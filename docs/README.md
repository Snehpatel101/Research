# Ensemble Price Prediction Pipeline - Documentation

**Status**: Phase 1 Complete (Production Ready)
**Version**: 1.0
**Last Updated**: December 23, 2025

---

## Quick Links

- **New to the project?** Start with [Quickstart Guide](getting-started/QUICKSTART.md)
- **Running the pipeline?** See [Pipeline CLI Reference](getting-started/PIPELINE_CLI.md)
- **Understanding the system?** Read [Architecture Overview](reference/ARCHITECTURE.md)
- **What's next?** Check [Development Status](development/STATUS.md) and [Roadmap](development/ROADMAP.md)

---

## Documentation Structure

### Getting Started
Step-by-step guides to get you running quickly.

- [**QUICKSTART.md**](getting-started/QUICKSTART.md) - 15-minute setup and first pipeline run
- [**PIPELINE_CLI.md**](getting-started/PIPELINE_CLI.md) - Complete CLI reference with examples

### Guides
In-depth explanations of key pipeline concepts.

- [**LABELING.md**](guides/LABELING.md) - Triple-barrier labeling methodology
- [**VALIDATION.md**](guides/VALIDATION.md) - Data quality checks and validation rules
- [**LOOKAHEAD_PREVENTION.md**](guides/LOOKAHEAD_PREVENTION.md) - Preventing label leakage (purge/embargo)

### Reference
Technical specifications and detailed documentation.

- [**ARCHITECTURE.md**](reference/ARCHITECTURE.md) - System architecture, pipeline stages, implementation details
- [**FEATURES.md**](reference/FEATURES.md) - Complete feature catalog (50+ indicators)
- [**SLIPPAGE.md**](reference/SLIPPAGE.md) - Transaction cost modeling and GA penalties

### Phases
Frozen specifications for the 5-phase project plan.

- [**PHASE_1.md**](phases/PHASE_1.md) - Data Preparation & Labeling (Complete)
- [**PHASE_2.md**](phases/PHASE_2.md) - Training Base Models (Next)
- [**PHASE_3.md**](phases/PHASE_3.md) - Cross-Validation & OOS Predictions
- [**PHASE_4.md**](phases/PHASE_4.md) - Train Ensemble Meta-Learner
- [**PHASE_5.md**](phases/PHASE_5.md) - Full Integration & Final Test

### Development
Current status and future plans.

- [**STATUS.md**](development/STATUS.md) - Phase 1 completion checklist and Phase 2 readiness
- [**ROADMAP.md**](development/ROADMAP.md) - Phase 2 implementation plan and architecture

---

## Phase 1 Summary

### What We Built
A production-ready data preparation and labeling pipeline that:

1. Ingests raw 1-minute OHLCV data (MES, MGC futures)
2. Resamples to 5-minute bars with proper forward-fill
3. Computes 50+ technical features with GA optimization
4. Applies symbol-specific triple-barrier labeling
5. Creates train/val/test splits with proper purge/embargo
6. Generates quality-weighted labels (0.5x-1.5x)
7. Validates data integrity and prevents look-ahead bias

### Key Achievements
- **Zero critical bugs** - All leakage issues fixed
- **Zero runtime blockers** - Pipeline executes successfully
- **Modular architecture** - Clean stage separation, easy to extend
- **Comprehensive validation** - 40% test coverage, 19/19 tests passing
- **Symbol-specific config** - MES asymmetric (1.5:1.0), MGC symmetric barriers
- **Performance optimized** - Numba JIT (10x speedup on labeling)

### Metrics
- **Codebase**: ~10,063 lines of Python
- **Pipeline Stages**: 8 (ingest → validation)
- **Features**: 50+ technical indicators
- **Symbols**: MES, MGC
- **Horizons**: H5, H10, H15, H20 (bars)
- **Data Split**: 70/15/15 (train/val/test)
- **Purge**: 60 bars (prevents label leakage)
- **Embargo**: 288 bars (~1 trading day)

---

## Quick Command Reference

```bash
# Run full pipeline for both symbols
./pipeline run --symbols MES,MGC

# Run specific stages
./pipeline run --symbols MES --stages stage1,stage2,stage3

# Resume from specific stage
./pipeline rerun <run_id> --from stage7

# Check pipeline status
./pipeline status <run_id>

# Validate configuration
./pipeline validate

# Run tests
pytest tests/phase_1_tests/
```

---

## Configuration Files

Key configuration locations:

```
/home/jake/Desktop/Research/
├── src/
│   ├── config/
│   │   ├── barriers_config.py     # Triple-barrier parameters
│   │   ├── features.py            # Feature selection thresholds
│   │   ├── feature_sets.py        # Feature set definitions
│   │   ├── labeling_config.py     # Labeling configuration
│   │   └── validation.py          # Validation rules
│   ├── horizon_config.py          # Horizon definitions (H5-H20)
│   ├── pipeline_config.py         # Pipeline-wide settings
│   └── presets.py                 # Symbol-specific presets
├── config/
│   └── ga_results/                # GA optimization results
└── data/
    ├── splits/                    # Train/val/test splits
    └── raw/validated/             # Validated metadata
```

---

## Development Guidelines

### Engineering Principles
1. **Modularity**: No monoliths, clear separation of concerns
2. **File limits**: 650 lines max per file
3. **Fail fast**: Validate at boundaries, explicit errors
4. **Simplicity**: Less code is better, avoid premature abstraction
5. **Testing**: Every module ships with tests proving the contract
6. **Delete legacy**: If unused, remove it (Git is the archive)

### Adding New Features
1. Define in `src/config/feature_sets.py`
2. Implement in `src/stages/features/engineer.py`
3. Add tests in `tests/phase_1_tests/stages/test_stage3_feature_engineering_core.py`
4. Run validation: `pytest tests/phase_1_tests/`

### Adding New Pipeline Stages
1. Create stage file in `src/pipeline/stages/`
2. Register in `src/pipeline/stage_registry.py`
3. Add to preset in `src/presets.py`
4. Add tests in `tests/phase_1_tests/pipeline/`

---

## Expected Performance (Phase 1 Baseline)

Based on GA-optimized barriers and transaction cost penalties:

| Horizon | Expected Sharpe | Win Rate | Max Drawdown |
|---------|----------------|----------|--------------|
| H5      | 0.3 - 0.8      | 45-50%   | 10-25%       |
| H10     | 0.4 - 0.9      | 46-52%   | 9-22%        |
| H15     | 0.5 - 1.0      | 47-53%   | 8-20%        |
| H20     | 0.5 - 1.2      | 48-55%   | 8-18%        |

Note: These are baseline expectations from triple-barrier labeling alone. Phase 2 ML models should improve these metrics significantly.

---

## Support & Troubleshooting

### Common Issues
- **Import errors**: Check virtual environment is activated
- **Data validation failures**: Verify input data schema matches expectations
- **GA optimization slow**: Reduce population size or generations in config
- **Memory issues**: Process symbols individually or reduce feature count

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
./pipeline run --symbols MES --verbose

# Check pipeline state
cat runs/<run_id>/artifacts/pipeline_state.json
```

### Getting Help
1. Check relevant guide in `docs/guides/`
2. Review architecture docs in `docs/reference/`
3. Search issues in git history
4. Review test cases for examples

---

## What's Next?

Phase 2 will add ML model training capabilities:

1. **TimeSeriesDataset** - Proper sequence handling for LSTM/Transformer models
2. **Base Models** - LightGBM, XGBoost, Random Forest, LSTM
3. **Training Pipeline** - Cross-validation, hyperparameter tuning
4. **Model Registry** - Track experiments, versions, metrics
5. **Prediction Pipeline** - Out-of-sample predictions for ensemble

See [Development Roadmap](development/ROADMAP.md) for details.

---

## License & Attribution

This is a research project for ensemble price prediction on OHLCV futures data.
All code follows clean architecture principles with modular, testable components.

**Built with**: Python 3.11+, Polars, NumPy, Numba, DEAP, pytest
