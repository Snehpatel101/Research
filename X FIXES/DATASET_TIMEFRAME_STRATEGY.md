# “Nine Datasets Across Different Timeframes” — What It Should Mean

The phrase “produce nine datasets across different timeframes” can mean different things architecturally. Choosing the meaning determines how Phase 1/2/3 must be structured.

## Updated target understanding (your clarification)

You want a *market-data fanout* followed by *per-model dataset views*:

1. Start from **one uploaded dataset**.
2. Materialize **multiple timeframe datasets** (the “market data store”).
3. Each model chooses, per run (configurable):
   - **Single-TF only** (train on one timeframe, no MTF features)
   - **Single-TF + higher-TF indicators** (train on one timeframe, enrich with indicators derived from other timeframes)
   - **Multi-TF ingestion** (ingest multiple timeframes simultaneously, e.g., 3 timeframes as separate streams/tensors)

This is closer to a hybrid of “materialize TF datasets” *and* “dataset views”, where the materialized TF datasets serve as reusable primitives.

## Interpretation A (most literal): 9 independent base-timeframe datasets

For each timeframe in `{1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h}` produce:

- cleaned OHLCV at that base TF,
- features computed at that base TF,
- labels computed on that base TF (horizons must be time-consistent or explicitly redefined),
- splits + scaling computed on that base TF,
- final datasets saved and versioned.

### Implications

- Label horizons currently mean “bars”; across TFs that changes the time horizon unless scaled.
- Purge/embargo bars must be timeframe-aware (utilities exist, but orchestration does not).
- Storage cost: 9× feature/label artifacts unless you cache intermediate computations.

### Current blockers in code

- Phase 1 config has a single `target_timeframe` (one TF per run).
- Stage 2 can only produce one TF output (even though `clean_symbol_data_multi_timeframe()` exists).
- Stage 7.6 builds datasets by `(feature_set, horizon)`, not by `(timeframe, feature_set, horizon)`.

## Interpretation B (canonical 1m + “dataset views” per model)

Instead of materializing 9 full datasets, treat the canonical dataset as **1m** (or another canonical TF) and generate “views”:

- resample OHLCV/features on-the-fly to the model’s desired TF,
- generate model-specific training tensors via adapters (2D/3D/4D),
- optionally cache per-view outputs.

### Implications

- Requires a clean, explicit contract for “dataset view”:
  - `base_tf`, `label_policy`, `feature_set`, `mtf_strategy`, `sequence_window_policy`, `scaling_policy`
- Requires training to ask for a view, not a directory path.

### Current blockers in code

- `TimeSeriesDataContainer` primarily loads pre-materialized split files from a directory and infers columns; it is not a “view builder” abstraction.
- MTF generator exists, but the pipeline uses it as feature-column augmentation, not a general view system.

## Interpretation C (recommended for your goal): Materialized TF store + per-model view selection

This matches your “robust pipeline / different configurations every run” requirement.

### Conceptual architecture

- **MarketDataStore** (outputs of Phase 1):
  - timeframes: `{1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h}`
  - for each timeframe, a consistent contract: `clean OHLCV`, optional `features`, optional `labels`, optional `splits/scaling`
- **ModelDatasetView** (used by training):
  - `primary_timeframe` (which TF the model trains “on”)
  - `mtf_strategy`:
    - `none` (single-TF only)
    - `indicators` (add indicators computed from selected other TFs)
    - `ingestion` (multi-stream input, e.g., 3 TFs simultaneously)
  - `feature_set` (per model family, already partially supported via feature set definitions)

### Why this is the right fit

- You get deterministic, reusable “building block” datasets (timeframe fanout).
- You still allow per-model flexibility (single TF vs MTF indicators vs multi-stream ingestion).
- Ensembles become straightforward: base models can deliberately use *different* timeframes/strategies to reduce error correlation.

### Current blockers in code (specific)

- Fanout is not orchestrated (Stage 2 produces one TF; multi-TF cleaner exists but isn’t used).
- MTF indicator generation exists, but it currently augments one base dataset and is not expressed as a per-model view contract.
- Multi-stream ingestion needs a clear, persisted representation:
  - `MultiResolution4DAdapter` exists, but there is no canonical pipeline output contract that guarantees the required per-timeframe columns/arrays are present for it.

## Recommended decision point

Pick **one** of the above as the canonical meaning of “nine datasets”.

- If you want *maximum reproducibility and simplest training*, pick Interpretation A.
- If you want *maximum flexibility and minimal storage*, pick Interpretation B.
- If you want **your described behavior** (fanout + per-model selection), pick Interpretation C.

Then enforce that decision in:

- config vocabulary,
- pipeline stage outputs,
- dataset container/adapters interface,
- docs/quickstarts.
