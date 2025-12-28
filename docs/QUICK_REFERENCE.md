# Quick Reference

This is the “what to run” page for the repo as it exists today.

## Notebook-first workflow (recommended)

- Notebook entrypoint: `notebooks/ML_Pipeline.ipynb`
- Notebook docs: `docs/notebook/README.md`
- Colab setup: `docs/notebook/COLAB_SETUP.md` (or `docs/COLAB_GUIDE.md`)

## Phase 1: Data pipeline (CLI)

```bash
# Requires data in data/raw/ (e.g. SI_1m.parquet)
./pipeline run --symbols SI

# Inspect run metadata / logs
./pipeline status <run_id>
tail -f runs/<run_id>/logs/pipeline.log
```

**Outputs (current behavior):**
- Run metadata: `runs/<run_id>/...`
- Data artifacts written under shared `data/` paths (e.g. `data/splits/scaled/train_scaled.parquet`)

## Phase 2: Training

```bash
python scripts/train_model.py --list-models
python scripts/train_model.py --model xgboost --horizon 20
```

Training outputs are written under `experiments/runs/<training_run_id>/`.

## Phase 3: Cross-validation (purged K-fold)

```bash
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
```

CV results are written under `--output-dir` (default: `data/stacking/`), including:
- `cv_results.json`
- `tuned_params/*.json` (when tuning enabled)
- `stacking/*.parquet` (OOF datasets)

## Phase 4: Ensembles

Ensembles are model types in the registry (voting/stacking/blending):

```bash
python scripts/train_model.py --model voting --horizon 20
python scripts/train_model.py --model stacking --horizon 20
python scripts/train_model.py --model blending --horizon 20
```

## Phase 5: Inference / serving

Serving and batch inference operate on **bundle directories**:

```bash
python scripts/serve_model.py --bundle /path/to/bundle --port 8080
python scripts/batch_inference.py --bundle /path/to/bundle --input /path/to/input.parquet --output /path/to/preds.parquet
```

See `docs/phases/PHASE_5.md`.

