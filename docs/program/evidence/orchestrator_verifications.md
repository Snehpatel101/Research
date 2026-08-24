# Orchestrator's own verifications (independent of subagents)

These are checks I ran myself to confirm or correct subagent claims.
Anything here is OBSERVED unless stated otherwise.

## V1. Model roster: 16 base learners, not 12 — CONFIRMS Agent 2

`src/core/constants.py:79-99`, `assert len(ALL_MODELS) == 23`.
Families: boosting(3) + classical(3) + neural(7) + transformer(3) = **16 base
learners**; plus ensemble(3) + meta_learner(4) = 23 total entries.

CLAUDE.md's "All 12 models are production-ready" table is an **undercount**. It
omits `random_forest`, `logistic`, `svm`, `transformer`.

**Mission impact:** the cheap baselines required to prove ensembles beat their
constituents (logistic regression, random forest) ALREADY EXIST. That work is
promotion + wiring, not implementation.

## V2. predict_proba absence is cosmetic, not structural — CORRECTS Agent 2

Agent 2 flagged that `predict_proba` exists on classical/ensemble/meta models
but on none of the 3 boosting or 10 neural models, framing it as a roster-
splitting hazard.

`src/core/interfaces.py:156-159` — `PredictionResult` declares
`class_predictions`, `class_probabilities`, `confidence` as **required**
(non-default) fields. Every model's `predict()` returns this type.

So probabilities DO flow uniformly from all 16 base learners via the canonical
path. Soft voting / stacking / calibration are NOT blocked.

The residual risk is real but narrow: any code that duck-types
`hasattr(model, "predict_proba")` will silently split the roster. That is a
lint/protocol issue, not an architectural one.

**Downgrade severity: HIGH -> LOW.**

## V3. Four-registry "drift" is mostly NOT real — CORRECTS Agent 2

Agent 2 claimed observed drift across `MODEL_CONTRACTS`,
`MODEL_DATA_REQUIREMENTS`, `MODEL_FEATURE_STRATEGIES`, `HYPERPARAMETER_SPACES`.

Cross-checked all four by parsing their literal keys against the 16 base
learners:

| Registry | n | missing base learners | extras |
|---|---|---|---|
| MODEL_CONTRACTS | 23 | none | 7 ensemble/meta (legitimate) |
| MODEL_DATA_REQUIREMENTS | 24 | none | 7 ensemble/meta + **`mlp` (phantom)** |
| MODEL_FEATURE_STRATEGIES | 16 | none | none |
| HYPERPARAMETER_SPACES | 23 | none | 7 ensemble/meta (legitimate) |

**All 16 base learners are present in all four registries.** The only genuine
phantom is `mlp` in MODEL_DATA_REQUIREMENTS (no backing class).

The valid concern survives in weaker form: adding a model requires edits at ~6
declaration sites. That is **duplication of declaration**, which is a real
mission concern ("adding a model should require minimal platform work"), but it
is NOT "drift causing bugs".

**Reclassify: "registry drift (HIGH)" -> "declaration-site duplication (MEDIUM,
architectural)" + "one phantom `mlp` entry (LOW, trivial fix)".**

## V4. Environment

- No ML deps were installed in the container at session start. Built `.venv`
  via uv. `download.pytorch.org` is blocked by the agent proxy; PyPI works.
- Resolved: torch 2.13.0+cu130 (CPU-only at runtime, `cuda.is_available()`
  False), pandas **3.0.5**, numpy **2.4.6**, sklearn 1.9.0, xgboost 3.2.0,
  lightgbm 4.7.0, catboost 1.2.10, optuna 4.9.0, numba 0.67.0.
- pandas 3.0 / numpy 2.4 are well beyond the repo's `pandas>=2.1` target.
  Core imports succeed. Suite impact pending.
- Hardware: 4 CPUs, 15 GB RAM, ~30 GB disk, no GPU. This bounds the
  combination matrix; transformer/TFT runs must stay small.

## V5. Datasets available

| File | Rows | Index |
|---|---|---|
| MES_1m_1week.parquet | 6,825 | RangeIndex + `datetime` col |
| MES_1m_1month.parquet | 29,776 | RangeIndex + `datetime` col |
| _test_subset_1mo.parquet | 31,680 | **DatetimeIndex**, no datetime col |
| MES_1m / mes-1m_data_2020 | 350,464 | RangeIndex |
| MGC-1m_data_2020 | 355,250 | RangeIndex |
| MGC_1m | 690,000 | RangeIndex |
| MGC_1m_5year | 1,601,940 | RangeIndex |
| SI_1m_validated | 5,333,201 | RangeIndex + `symbol` |
| ng-1m_bk_2020 | 336,073 | RangeIndex |
| MCL-1m_data_2020 | 351,028 | RangeIndex |

Five distinct instruments (MES, MGC, SI, NG, MCL) => genuine cross-symbol
generalization tests are possible. Note the two index conventions in the wild —
a real portability hazard for any loader.

## V6. Heterogeneous OOF alignment is BROKEN — CONFIRMS Agent 3, upgraded to PROVEN

Reproduction: `docs/program/evidence/F1_oof_misalignment_repro.py` (committed).
Uses the real `SequenceAdapter` and the real `OOFAligner`.

- `sequence.py:234-236` computes CORRECT source-bar coords:
  `original_indices = df_indices[label_positions]`, starting at `seq_len-1`.
  Measured: `[59 60 61 62 63]` for seq_len=60.
- `oof_generation.py:383` emits POSITIONAL coords:
  `valid_indices = np.where(~np.isnan(oof_preds))[0]` over the model's OWN
  windowed array. Measured: `[0 1 2 3 4]`.
- `oof_generation.py:373` writes `"datetime": range(n_samples)`.
- `alignment.py` `OOFAligner._find_intersection_indices` set-intersects those
  positional integers as if they were a shared key.

Measured with two DELIBERATELY PERFECT models:
    agreement between two PERFECT models = 0.0%
    expected if aligned correctly        = 100.0%
    systematic shift                     = 59 bars

**Severity: CRITICAL.** Silent. No crash, no warning, plausible val_f1.
Invalidates CLAUDE.md Phase 57 ("cross-family ensembles ... working together")
and Phase 60 ("All 8 ensemble combinations now PASS") — those passed by not
crashing. Every cross-family ensemble metric ever reported is meaningless.

## V7. CLAUDE.md Phase 94 claim is FALSE — CONFIRMS Agent 4

CLAUDE.md Phase 94: "Cumulative features (OBV/VWAP/TWAP/cum_order_flow) reset
at session boundaries via `session_cumsum()`".

    $ grep -rn "session_cumsum" src/     ->  (no matches)

The function does not exist. Documented work that was never done or was
deleted without correcting the doc.

## V8. No naive baseline exists anywhere — CONFIRMS Agent 4

    $ grep -rniE "class (Dummy|Naive|Majority|Baseline).*(Model|Classifier)" src/
      -> (no matches)

**Mission-critical.** The mission requires proving ensembles beat their
constituents AND simple baselines. There is currently nothing to compare
against. Note this is cheap to fix and `logistic`/`random_forest` already
exist as strong-ish baselines (see V1).

## V9. 3D/4D models have NO test-set evaluation — CONFIRMS Agent 4

`src/models/training/trainer.py:1104`:
    "test_metrics": None,  # Test eval not supported in run_prepared yet

10 of 16 base learners therefore have zero held-out generalization estimate.

## V10. MDA feature ranking runs on SHUFFLED rows — CONFIRMS Agent 4

`src/models/training/feature_selection.py:83-89` uses
`train_test_split(clean_df, train_size=mda_max_rows, stratify=..., random_state=42)`.
`shuffle` is not passed => defaults to **True**. Temporal order is destroyed
before positional purge/embargo is applied, making the purge meaningless and
systematically over-ranking autocorrelated features.

CLAUDE.md Phase 103 claims "Strided temporal subsampling (replaces random
stratified that destroyed temporal structure)". That fix is NOT present at
this call site.

## V11. Financial report prices are head-sliced — CONFIRMS Agent 4

`feature_selection.py:742`: `prices_slice = prices[: len(oof.get_class_predictions())]`
head-slices the full-df close array, but `filter_invalid_labels()` has already
dropped rows from BOTH head and tail. Every trade in the reported equity curve
is attached to the wrong bar. Computed on the TRAIN split, never on test.

## V12. Capability truth is DUPLICATED and DRIFTED — CONFIRMS Agent 5

Constructed all 23 registered models and compared class properties against
`MODEL_CONTRACTS`. All 23 construct successfully with zero args (good).

Drift found, exactly as Agent 5 predicted (4 items):

| model | field | class | contract |
|---|---|---|---|
| patchtst | model_family | neural | transformer |
| itransformer | model_family | neural | transformer |
| transformer | model_family | neural | transformer |
| mlp_meta | requires_scaling | False | True |

Family drift is NOT cosmetic — it selects different feature-set defaults.

Also OBSERVED: `tft` is `SEQUENCE_3D` with `requires_4d=False`, contradicting
the CLAUDE.md 4D table. Only `patchtst` and `itransformer` are truly 4D.

## V13. The registry is CLOSED BY ASSERTION — CONFIRMS Agent 5

`src/core/constants.py:100`:
    assert len(ALL_MODELS) == 23, f"Expected 23 models, got {len(ALL_MODELS)}"

Import-time. Adding a 24th model crashes the process on import. For a system
whose stated goal is "adding a new model should mean implement + declare +
register", this is a literal, hard-coded ceiling.

## V14. Correction to my own earlier check

`ModelRegistry.list_models()` returns a `family -> [names]` mapping, not a
list of names. My first drift script iterated its KEYS and so compared 6
family strings. That was my error, not a repo defect. Re-run corrected;
results above.
