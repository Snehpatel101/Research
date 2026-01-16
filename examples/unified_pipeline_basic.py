"""
Basic example of using the unified ML pipeline.

This example demonstrates how to use the MLPipeline class to run
the complete ML workflow from raw data to trained models.
"""

from pathlib import Path
from src.ml_pipeline import MLPipeline, MLConfig, ModelConfig


def basic_example():
    """Run basic pipeline with single model."""
    config = MLConfig(
        symbol="MES",
        horizons=[20],
        models=["xgboost"],
        training_mode="standard",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    pipeline = MLPipeline(config)

    print(f"Starting pipeline (run_id={config.run_id})")
    print(f"Symbol: {config.symbol}")
    print(f"Horizons: {config.horizons}")
    print(f"Models: {[m.name for m in config.models]}")

    # Note: This will fail if you don't have actual MES data
    # results = pipeline.run()
    # print(f"Pipeline complete! Results saved to: {results['output_dir']}")


def advanced_example():
    """Run advanced pipeline with multiple models and ensemble."""
    config = MLConfig(
        symbol="MES",
        horizons=[20],
        models=[
            ModelConfig(
                name="xgboost",
                timeframe="15min",
                optimize_features=True,
                feature_opt_trials=30,
            ),
            ModelConfig(
                name="lstm",
                timeframe="5min",
                optimize_features=True,
                sequence_length=60,
            ),
            ModelConfig(
                name="patchtst",
                timeframe="1min",
            ),
        ],
        build_ensemble=True,
        meta_learner="ridge_meta",
        training_mode="standard",
        evaluation_methods=["cv"],
    )

    pipeline = MLPipeline(config)

    print(f"Starting advanced pipeline (run_id={config.run_id})")
    print(f"Training {len(config.models)} models:")
    for model_cfg in config.models:
        print(f"  - {model_cfg.name} (timeframe={model_cfg.timeframe})")

    # Note: This will fail if you don't have actual MES data
    # results = pipeline.run()


def phase_by_phase_example():
    """Run pipeline phase by phase with checkpointing."""
    config = MLConfig(
        symbol="MES",
        horizons=[20],
        models=["xgboost"],
    )

    pipeline = MLPipeline(config)

    # Phase 1-5: Data pipeline
    print("Running data pipeline...")
    # data_results = pipeline.run_data()
    # print(f"Data pipeline complete!")
    # print(f"Labeled data paths: {data_results['labeled_data_paths']}")

    # Phase 6: Training
    print("Running training...")
    # training_results = pipeline.run_training()
    # print(f"Training complete!")
    # print(f"Model paths: {training_results['model_paths']}")

    # Phase 7: Evaluation (if configured)
    if config.evaluation_methods:
        print("Running evaluation...")
        # eval_results = pipeline.run_evaluation()
        # print(f"Evaluation complete!")


def resume_example():
    """Resume pipeline from previous run."""
    # Load config from previous run
    run_id = "20260116_120000_abc123"

    config = MLConfig(
        symbol="MES",
        horizons=[20],
        models=["xgboost"],
        run_id=run_id,  # Use existing run_id
    )

    pipeline = MLPipeline(config)

    print(f"Resuming pipeline from run_id={run_id}")
    print(f"Current status: {pipeline.state.status}")

    # Resume from last checkpoint
    # results = pipeline.resume()
    # print(f"Resume complete!")


def yaml_config_example():
    """Load config from YAML file."""
    # Create YAML config
    yaml_content = """
symbol: MES
horizons: [20]
models:
  - name: xgboost
    timeframe: 15min
    optimize_features: true
  - name: lstm
    timeframe: 5min
    optimize_features: true
training_mode: standard
build_ensemble: true
meta_learner: ridge_meta
"""

    # Save to file
    config_path = Path("config_example.yaml")
    config_path.write_text(yaml_content)

    # Load and run
    pipeline = MLPipeline(config_path)
    print(f"Loaded config from {config_path}")

    # Clean up
    config_path.unlink()


if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED ML PIPELINE - BASIC EXAMPLES")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Example 1: Basic single model")
    print("=" * 70)
    basic_example()

    print("\n" + "=" * 70)
    print("Example 2: Advanced multi-model with ensemble")
    print("=" * 70)
    advanced_example()

    print("\n" + "=" * 70)
    print("Example 3: Phase-by-phase execution")
    print("=" * 70)
    phase_by_phase_example()

    print("\n" + "=" * 70)
    print("Example 4: Resume from checkpoint")
    print("=" * 70)
    resume_example()

    print("\n" + "=" * 70)
    print("Example 5: YAML config")
    print("=" * 70)
    yaml_config_example()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNote: Actual pipeline.run() calls are commented out")
    print("because they require real MES data to be available.")
    print("\nTo run with real data:")
    print("  1. Ensure data/raw/MES_1m.parquet exists")
    print("  2. Uncomment the pipeline.run() calls")
    print("  3. Run this script: python examples/unified_pipeline_basic.py")
