# Remediation Roadmap (From “Split-Brain” → Robust ML Factory)

This roadmap is intentionally outcome-driven: each phase ends with verifiable acceptance criteria aligned to the target workflow.

## Phase 0 — Repo hygiene + trust reset (1–2 days)

**Goal:** Make the repo readable and predictable again.

- Remove tracked generated artifacts (`__pycache__/`, `*.pyc`, caches, in-repo `.venv/`).
- Ensure `.gitignore` prevents reintroduction.
- Add a root `README.md` (or fix `pyproject.toml` to point to the actual readme).

**Acceptance criteria**
- `git status` is clean after running tests/linters (no new cache noise).
- Root quickstart exists and points to the single authoritative run commands.

## Phase 1 — Define the factory’s canonical contracts (1–3 days)

**Goal:** One vocabulary + one set of rules for timeframes, paths, and config precedence.

- Choose the canonical timeframe vocabulary (`1min/5min/.../1h` vs `60min`) and implement aliases consistently.
- Decide the meaning of “nine datasets”:
  - Materialized datasets per TF (Interpretation A), or
  - Canonical dataset + dataset views (Interpretation B).
- Publish config precedence: `defaults < YAML < CLI`.
- Publish artifact layout rules:
  - pipeline run outputs,
  - training run outputs,
  - bundles.

**Acceptance criteria**
- One “timeframe registry” is the only authority used for validation and conversion.
- One short doc explains where artifacts live and how to locate them by `run_id`.

## Phase 2 — Make Phase 1 truly configurable (1–3 days)

**Goal:** CLI knobs actually change behavior deterministically.

- Wire `feature_toggles` into Stage 3 feature engineering.
- Wire `barrier_overrides` precedence into Stage 4/6 labeling.
- Wire `scaler_type` into Stage 7.5 scaling (and reconcile with feature-set recommendations).

**Acceptance criteria**
- Running the pipeline twice with different toggles produces different, documented feature sets.
- Overriding labeling/scaling via CLI is reflected in persisted manifests/config outputs.

## Phase 3 — Implement “nine datasets” (3–10 days, depends on chosen interpretation)

### If Interpretation A (materialized per TF)

- Add multi-timeframe orchestration:
  - Stage 2 produces multiple cleaned TF datasets.
  - Downstream stages either:
    - run per TF, or
    - operate on a canonical TF and generate derived views with explicit mapping.
- Ensure horizon scaling is time-consistent per TF.
- Store artifacts under `runs/{run_id}/data/{timeframe}/...` (or equivalent) so the TF dimension is explicit.

### If Interpretation B (views)

- Introduce a `DatasetView` abstraction used by training:
  - view = (base_tf, label_policy, feature_set, mtf_strategy, window_policy, scaling_policy)
- Add caching policy for views (optional).

**Acceptance criteria**
- After one upload and one command, the system can produce and enumerate 9 TF datasets (materialized or view-cached).
- A single model can be trained against any of the 9 TF datasets without manual path edits.

### If Interpretation C (recommended for your clarified goal): TF store + per-model view selection

This matches: “one dataset → many timeframe datasets → each model chooses single TF / MTF indicators / multi-TF ingestion (e.g., 3 TF streams) per run”.

- Materialize a **MarketDataStore** for the selected timeframes (fanout).
- Define a **ModelDatasetView** contract used by training:
  - `primary_timeframe`
  - `mtf_strategy: none | indicators | ingestion`
  - `mtf_timeframes` (which higher/lower TFs participate)
  - `feature_set` (per-model or per-family)
- Ensure the dataset builder can create:
  - 2D tabular views for boosting/classical,
  - 3D sequence views for recurrent/conv/transformers,
  - multi-stream / 4D views for multi-TF ingestion models.

**Acceptance criteria**
- One command produces the configured timeframe fanout and a manifest enumerating available TF datasets.
- A training command can specify per-model view config (TF + strategy) and trains without manual path edits.

## Phase 4 — Connect training to pipeline runs (2–5 days)

**Goal:** “Run pipeline → train model/ensemble” becomes one coherent story.

- Create one authoritative training entrypoint (CLI subcommand preferred).
- Make training accept a pipeline `run_id` (not a raw path) and resolve correct dataset locations internally.
- Link artifacts:
  - training run references the pipeline run manifest and config,
  - bundles reference training run + preprocessing graph config.

**Acceptance criteria**
- `pipeline run ...` outputs a `run_id`.
- `pipeline train --run-id ... --model xgboost --horizon 20` works without manual paths.

## Phase 5 — Heterogeneous ensembles as first-class (3–7 days)

**Goal:** “ensemble from the same market data” works across model families.

- Standardize OOF generation for both 2D and 3D models.
- Enforce compatibility rules + deterministic stacking dataset construction.
- Persist OOF artifacts and meta-learner training artifacts.

**Acceptance criteria**
- A heterogeneous stacking run can be executed with a single command and produces:
  - base model runs,
  - OOF predictions,
  - meta-learner run,
  - consolidated evaluation + bundle(s).

## Phase 6 — End-to-end validation (ongoing)

**Goal:** Ensure the factory stays correct while evolving.

- Add (or repair) one end-to-end integration test for:
  - pipeline run on small synthetic OHLCV,
  - training one model,
  - producing a bundle,
  - running inference.

**Acceptance criteria**
- CI (or local test command) can validate the “happy path” end-to-end on a small dataset.
