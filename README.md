# Ensemble Price Prediction Pipeline

A comprehensive ML pipeline for financial price prediction using ensemble methods with triple-barrier labeling.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
./pipeline run --symbols MES,MGC

# Check status
./pipeline status <run_id>
```

## Project Structure

```
research/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pipeline                  # CLI wrapper script
│
├── src/                      # Source code
│   ├── pipeline_cli.py       # CLI interface
│   ├── pipeline_runner.py    # Pipeline orchestrator
│   ├── pipeline_config.py    # Configuration management
│   ├── manifest.py           # Data versioning
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── data_cleaning.py
│   └── stages/               # Pipeline stages
│       ├── stage1_ingest.py
│       ├── stage2_clean.py
│       ├── stage3_features.py
│       ├── stage4_labeling.py
│       ├── stage5_ga_optimize.py
│       ├── stage6_final_labels.py
│       ├── stage7_splits.py
│       └── stage8_validate.py
│
├── data/                     # Data directory
│   ├── raw/                  # Raw 1-minute data (parquet)
│   └── splits/               # Train/val/test splits
│
├── docs/                     # Documentation
│   ├── phases/               # Phase documentation (1-5)
│   ├── guides/               # User guides & quickstarts
│   └── reference/            # Technical reference docs
│
├── tests/                    # Test files
│   ├── test_pipeline.py
│   ├── test_pipeline_system.py
│   ├── test_stages.py
│   └── verify_modules.py
│
├── scripts/                  # Utility scripts
│   └── verify_installation.sh
│
├── notebooks/                # Jupyter notebooks
├── reports/                  # Generated reports
└── results/                  # Pipeline results
```

## Pipeline Stages

1. **Ingest** - Load and validate raw data
2. **Clean** - Resample 1-min to 5-min bars
3. **Features** - Generate 50+ technical indicators
4. **Labeling** - Apply triple-barrier labeling
5. **GA Optimize** - Genetic algorithm optimization
6. **Final Labels** - Generate final labels
7. **Splits** - Create train/val/test splits with purging
8. **Validate** - Validate pipeline output

## CLI Commands

| Command | Description |
|---------|-------------|
| `./pipeline run` | Execute complete pipeline |
| `./pipeline rerun <id>` | Resume from failed stage |
| `./pipeline status <id>` | Check pipeline status |
| `./pipeline validate` | Validate configuration |
| `./pipeline list-runs` | List all pipeline runs |
| `./pipeline compare <id1> <id2>` | Compare two runs |
| `./pipeline clean <id>` | Delete a run |

Run `./pipeline --help` for detailed options.

## Configuration

```bash
# Run with custom parameters
./pipeline run \
  --symbols MES,MGC,MNQ \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --horizons 1,5,20 \
  --description "My experiment"
```

## Python API

```python
from src.pipeline_config import PipelineConfig, create_default_config
from src.pipeline_runner import PipelineRunner

# Create configuration
config = create_default_config(
    symbols=['MES', 'MGC'],
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Run pipeline
runner = PipelineRunner(config)
runner.run()
```

## Documentation

- **[Pipeline CLI Guide](docs/guides/PIPELINE_CLI_GUIDE.md)** - Complete CLI reference
- **[Quick Reference](docs/guides/PIPELINE_QUICK_REFERENCE.md)** - Command cheat sheet
- **[Installation Guide](docs/guides/INSTALLATION_SUMMARY.md)** - Setup instructions
- **[Labeling Quickstart](docs/guides/LABELING_QUICKSTART.md)** - Labeling overview

### Phase Documentation

- [Phase 1: Data Preparation](docs/phases/PHASE_1_Data_Preparation_and_Labeling.md)
- [Phase 2: Training Base Models](docs/phases/PHASE_2_Training_Base_Models.md)
- [Phase 3: Cross-Validation](docs/phases/PHASE_3_Cross_Validation_OOS_Predictions.md)
- [Phase 4: Ensemble Meta-Learner](docs/phases/PHASE_4_Train_Ensemble_Meta_Learner.md)
- [Phase 5: Full Integration](docs/phases/PHASE_5_Full_Integration_Final_Test.md)

## Testing

```bash
# Run tests
python -m pytest tests/

# Or run individual test files
python tests/test_pipeline_system.py
```

## License

Part of the Ensemble Price Prediction System.
