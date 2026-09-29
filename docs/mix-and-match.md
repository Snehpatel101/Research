# Mix and match

Any subset of the base models can be trained together, in any training mode,
stacked by any meta-learner. Nothing else in the config changes when you swap
one for another.

```python
cfg.training.models = ["xgboost", "lstm", "patchtst"]   # any subset, any ranks
cfg.training.meta_learner = "voting_meta"               # any of the five
cfg.training.training_mode = "walk_forward"             # any of the four
```

```bash
python -m src.cli run -d bars.parquet -m xgboost,lstm,patchtst \
    --build-ensemble --meta-learner voting_meta --training-mode walk_forward -h 5
```

## The building blocks

### Base models

Each model declares a **contract** (`src/core/contracts/model_contract.py`):
the input rank it needs, its window length, how many features it may keep and
how its inputs are scaled. The data adapters build exactly that input from the
same labeled feature frame.

| Model | Architecture | Input | Window (bars) | Max features | Scaling |
|---|---|---|---|---|---|
| `xgboost` | boosting | 2D table | — | 200 | none |
| `lightgbm` | boosting | 2D table | — | 200 | none |
| `catboost` | boosting | 2D table | — | 200 | none |
| `random_forest` | tree ensemble | 2D table | — | 150 | none |
| `logistic` | linear | 2D table | — | 100 | standard |
| `svm` | kernel | 2D table | — | 80 | standard |
| `lstm` | recurrent | 3D windows of the feature table | 60 | 150 | robust |
| `gru` | recurrent | 3D windows | 60 | 150 | robust |
| `tcn` | temporal convolution | 3D windows | 64 (receptive field 61) | 120 | robust |
| `inceptiontime` | convolution | 3D windows | 60 | 100 | robust |
| `resnet1d` | convolution | 3D windows | 60 | 100 | robust |
| `nbeats` | MLP (N-BEATS) | 3D windows | 60 | 20 | robust |
| `transformer` | attention encoder | 3D windows | 128 | 100 | standard |
| `tft` | Temporal Fusion Transformer | 3D windows | 60 | 100 | robust |
| `patchtst` | patch transformer | 4D: OHLCV windows at several timeframes (5 and 15 min) | 60 | 80 | standard |
| `itransformer` | inverted transformer | 4D: OHLCV windows at several timeframes (5 and 15 min) | 60 | 80 | robust |

`data.sequence.seq_len` overrides the window of every sequence model at once;
leave it `None` to use each contract's length. `python -m src.cli models <name>`
prints a model's default hyperparameters.

### Meta-learners

| Meta-learner | Combines the base models' OOF probabilities with |
|---|---|
| `ridge_meta` (default) | L2-regularized multinomial logistic regression |
| `xgboost_meta` | gradient-boosted trees |
| `mlp_meta` | a small MLP |
| `calibrated_meta` | a ridge classifier calibrated with time-series CV (isotonic / Platt) |
| `voting_meta` | a plain average — no fit, cannot overfit the OOF rows |

See [OOF stacking](concepts.md#out-of-fold-oof-stacking) for how the rows are
built, aligned and held out.

### Training modes

| Mode | What changes | Deployed artifact |
|---|---|---|
| `standard` | train / val / test split, purged k-fold OOF | one bundle per model + `EnsembleBundle` |
| `walk_forward` | out-of-sample signals from expanding or rolling windows | same as standard (walk-forward is the evaluation protocol) |
| `regime_aware` | one model per volatility / trend regime | `RegimeBundle` per model (per-bar routing) |
| `meta_labeling` | primary (`models[0]`) picks the side, a meta model filters bets | `MetaLabelingBundle` |

See [Training modes](concepts.md#training-modes) for the details.

## How ranks mix

A gradient-boosted tree predicts every bar; an LSTM with a 60-bar window has
nothing to say about the first 59; PatchTST needs history at several
timeframes. Stacking them naively (by row number) pairs one model's prediction
for bar *t* with another's for bar *t + 59* — a silent misalignment that looks
like a working ensemble.

ML Factory keys every out-of-fold prediction by the **source bar** it belongs
to. The ensemble trains on the bars every base model covers, and at inference
`EnsembleBundle.predict_from_raw` runs each base bundle on the raw bars,
intersects their prediction timestamps and only then applies the meta-learner.
The mix-and-match harness checks, for every combination, that stacking rows pair
each model's prediction with the label of the same bar.

## What is verified

`scripts/mix_match.py` runs full `MLFactory.run()` pipelines on synthetic bars —
features, labels, feature selection, training, OOF, ensemble, backtest, bundles,
deploy, reload, `predict_from_raw` — and fails a run unless:

- training succeeds and every model (and the ensemble) reports metrics;
- stacking rows pair every model's OOF prediction with the label of the same bar;
- the deployed bundle reproduces the trained model's validation probabilities;
- features recomputed from raw OHLCV equal the training features;
- the backtest produces metrics and the deploy artifact reloads and predicts.

```bash
python scripts/mix_match.py solo           # every model alone
python scripts/mix_match.py pairs          # every pair (120 ensembles)
python scripts/mix_match.py meta           # every meta-learner on xgboost + lstm + patchtst
python scripts/mix_match.py modes          # every training mode on the same trio
python scripts/mix_match.py modes-solo     # every model alone in each non-standard mode
python scripts/mix_match.py binary         # binary labels x meta-learners x modes
python scripts/mix_match.py all-in         # all 16 models in one ensemble
python scripts/mix_match.py custom xgboost,tcn --meta mlp_meta --mode regime_aware
python scripts/mix_match.py report         # regenerate docs/MIX_AND_MATCH.md
```

The latest results are in the [verification matrix](MIX_AND_MATCH.md).
`make matrix` runs everything (hours on a CPU; TFT and the transformer are the
slow ones).

## Adding a model

A new model plugs into every mode, meta-learner and bundle once it satisfies
the model interface and declares a contract.

1. **Implement `BaseModel`** (`src/models/base.py`) in the family's package,
   e.g. `src/models/neural/my_model.py`:

    ```python
    from src.models.base import BaseModel, PredictionResult, TrainingMetrics
    from src.models.common import map_classes_to_labels, map_labels_to_classes
    from src.models.registry import register


    @register(name="my_model", family="neural", description="My sequence model")
    class MyModel(BaseModel):
        # Required properties (plain class attributes work too)
        model_family = "neural"
        requires_scaling = True
        requires_sequences = True        # input (n, seq_len, n_features); 4D models
                                         # also set requires_4d = True

        def get_default_config(self) -> dict: ...
        def fit(self, X_train, y_train, X_val, y_val,
                sample_weights=None, config=None) -> TrainingMetrics: ...
        def predict(self, X) -> PredictionResult: ...
        def save(self, path) -> None: ...
        def load(self, path) -> None: ...
    ```

    The contract every implementation keeps:

    - Labels arrive as trading labels (−1/0/+1, or 0/1 in binary mode); map them
      with `map_labels_to_classes(y, self._n_classes)` and back with
      `map_classes_to_labels`. `class_probabilities` columns are in class order
      (short, neutral, long).
    - Use `sample_weights` when given (uniqueness weights).
    - Early-stop only on `X_val` / `y_val` — the pipeline decides what that is
      (a purged tail of the training rows inside CV folds, the validation split
      for the deployed model).
    - `save` / `load` must round-trip the fitted state; bundles rely on it.

2. **Register it**: import the module in the family's `__init__.py` so the
   `@register` decorator runs when `src.models` is imported, and add the name
   to `MODEL_FAMILIES` / `MODEL_TO_FAMILY` in `src/core/constants.py` (the
   pipeline config validates model names against them).

3. **Declare its contract** in `MODEL_CONTRACTS`
   (`src/core/contracts/model_contract.py`): `input_rank` (2D / 3D / 4D),
   `sequence_length`, `max_features`, `scaler_type`, `mtf_mode`.

4. **Optional — tuning**: add an Optuna search space in
   `src/validation/cv/param_spaces.py` (`get_param_space`).

5. **Verify**: add it to `BASE_MODELS` in `scripts/mix_match.py`, run
   `python scripts/mix_match.py custom my_model,xgboost` and then `solo` /
   `pairs`, and add unit tests under `tests/unit/models/`.
