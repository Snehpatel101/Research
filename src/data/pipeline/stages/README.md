# Pipeline Stages

Building blocks of the one ML Factory pipeline (`MLFactory`, `src/factory.py`).
Raw OHLCV bars go in; `MLFactory.prepare_data` runs them through `FeatureEngineer` and
the triple-barrier labeler (`src/data/labeling`), and the training orchestrator does the rest.

```
stages/
├── features/    FeatureEngineer + ~190 feature functions (Numba-accelerated)
├── mtf/         Multi-timeframe features (every MTF operation uses shift(1))
├── clean/       resample_ohlcv (shared by training and inference)
├── sessions/    CME trading calendar and session definitions
└── regime/      Market regime detection (volatility / trend / structure)
```

## Features (`features/`)

`FeatureEngineer.engineer_features(df, symbol=...)` generates momentum, volatility,
volume, trend, temporal, regime, wavelet, entropy and microstructure features, plus MTF
features when `enable_mtf=True`. `FeatureEngineer.to_spec()` records the exact recipe
so inference bundles replay the same transform (`PreprocessingGraph`).

## Multi-timeframe (`mtf/`)

`MTFFeatureGenerator` resamples to higher timeframes and joins them back with
`shift(1)`, so a bar only ever sees completed higher-timeframe bars.

## Clean (`clean/utils.py`)

`resample_ohlcv(df, timeframe)` resamples OHLCV with `closed="left", label="left"`.
`MLFactory` (`data.bar_timeframe`) and the inference preprocessing graph use the same function.

## Sessions (`sessions/`)

Session definitions and the CME holiday calendar used by the backtester's execution model.

## Regime (`regime/`)

Regime detectors used by `training_mode="regime_aware"` and regime-conditional evaluation.

## Running it

```bash
ml run -d data/mes_5min.parquet -m xgboost          # full pipeline
ml data -d data/mes_5min.parquet --horizons 5,20    # features + labels to parquet only
```

See the README for the full CLI.
