# Commands Reference

## Data Pipeline (Phase 1)

```bash
# Run full data pipeline for a symbol
./pipeline run --symbols MES
./pipeline run --symbols MGC

# Dry run (validate config without executing)
./pipeline run --symbols MES --dry-run

# Run specific stages
./pipeline run --symbols MES --stages ingest,clean,features
```

## Model Training (Phase 2)

```bash
# Train specific model
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lightgbm --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30

# List available models
python scripts/train_model.py --list-models

# Train ensemble (same-family models only!)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20
python scripts/train_model.py --model stacking --base-models lstm,gru,tcn --horizon 20
```

## Cross-Validation (Phase 3)

```bash
# Run CV with specific models
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Run CV with hyperparameter tuning
python scripts/run_cv.py --models xgboost --horizons 20 --tune

# Run all models
python scripts/run_cv.py --models all --horizons 5,10,15,20

# Walk-forward validation
python scripts/run_walk_forward.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test categories
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest tests/ -m "not slow" -v

# Run specific test file
pytest tests/models/test_boosting.py -v

# Run lookahead tests (IMPORTANT for leakage validation)
pytest tests/phase_1_tests/stages/test_lookahead_invariance.py -v
pytest tests/validation/test_lookahead.py -v
```

## Code Quality

```bash
# Format code
black src tests scripts

# Lint code
ruff check src tests scripts
ruff check --fix src tests scripts  # Auto-fix

# Type checking
mypy src

# Security scan
bandit -r src
```

## Inference & Serving

```bash
# Run batch inference
python scripts/batch_inference.py --model-path experiments/runs/latest

# Serve model via Flask
python scripts/serve_model.py --port 5000
```

## MCP Proxy (for Serena SSE Bridge)

```bash
# Start MCP proxy server
make mcp-start

# Stop MCP proxy
make mcp-stop

# Check status
make mcp-status

# Debug mode (foreground with logs)
make mcp-logs

# Configure Claude Code for SSE
make mcp-setup
```

## Git Commands

```bash
# Status and diff
git status
git diff
git diff --staged

# Commit (follow project guidelines)
git add -p  # Interactive staging
git commit -m "type: description"

# Branch management
git checkout -b feature/name
git push -u origin feature/name
```

## Utility Commands

```bash
# Check OHLCV data quality
python scripts/check_ohlcv.py

# Create holdout set
python scripts/create_holdout.py

# Diagnose label distribution
python scripts/diagnose_label_distribution.py

# Verify pipeline completion
python scripts/verify_pipeline_final.py
```

## Environment

```bash
# Install dependencies
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[all]"

# Activate virtual environment
source .venv/bin/activate
```
