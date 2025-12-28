# Phase 5: Inference, Serving, and Monitoring

## Status: PARTIAL

This repo contains the building blocks for inference/serving/monitoring, but it is not yet a single “train → bundle → deploy” one-command flow.

## What exists

- Bundle format + loader: `src/inference/bundle.py`
- Inference orchestration: `src/inference/pipeline.py`
- Batch inference: `src/inference/batch.py`, `scripts/batch_inference.py`
- HTTP serving (Flask optional): `src/inference/server.py`, `scripts/serve_model.py`
- Drift detection primitives: `src/monitoring/drift_detector.py`
- Alert handling primitives: `src/monitoring/alert_handler.py`

## What is not wired end-to-end

- A standard step that produces a **bundle** from `experiments/runs/<run_id>/...` training artifacts.
- A single “final test set evaluation” script (older docs referenced `scripts/evaluate_test.py`; it does not exist).
- A “test-set integrity audit log” mechanism (hash/access auditing).

## Serving and batch inference

Both serving and batch inference operate on a **bundle directory**:

```bash
# Start server
python scripts/serve_model.py --bundle /path/to/bundle --port 8080

# Batch inference to parquet
python scripts/batch_inference.py --bundle /path/to/bundle --input /path/to/input.parquet --output /path/to/preds.parquet
```

## Monitoring (drift detection)

Drift detection is implemented as reusable primitives (not a production daemon). You provide a stream of values (errors, features, prediction confidence) and handle emitted alerts:

- Detectors: `src/monitoring/drift_detector.py`
- Alert routing: `src/monitoring/alert_handler.py`

