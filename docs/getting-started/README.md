# Getting Started

Quick start guides for new users.

## Guides

| Document | Description | Time Required |
|----------|-------------|---------------|
| [Quickstart](QUICKSTART.md) | First pipeline run and model training | 30 minutes |
| [Pipeline CLI](PIPELINE_CLI.md) | Complete CLI reference | 15 minutes |

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Run data pipeline** (Phase 1)
   ```bash
   ./pipeline run --symbols MES
   ```

3. **Train a model** (Phase 2)
   ```bash
   python scripts/train_model.py --model xgboost --horizon 20
   ```

4. **Run cross-validation** (Phase 3)
   ```bash
   python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
   ```

## Next Steps

- Read [Model Integration Guide](../guides/MODEL_INTEGRATION_GUIDE.md) to add new models
- Review [Feature Engineering Guide](../guides/FEATURE_ENGINEERING_GUIDE.md) for feature strategies
- Check [Quick Reference](../QUICK_REFERENCE.md) for command cheatsheet

---

*Last Updated: 2025-12-30*
