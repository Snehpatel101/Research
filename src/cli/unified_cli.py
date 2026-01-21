"""
Unified CLI for ML Pipeline.

NOTE: This CLI uses MLPipeline which is deprecated in favor of MLFactory.
The CLI will emit deprecation warnings but remains functional for backward
compatibility. Future versions will migrate to MLFactory.

See src/factory.py for the canonical MLFactory entry point.

Provides simple command-line interface:
- ml run: Full pipeline (data + training + evaluation)
- ml data: Data pipeline only
- ml train: Training only
- ml evaluate: Evaluation only
- ml resume: Resume from checkpoint
- ml status: Show pipeline status
"""

import argparse
import sys
from pathlib import Path

from src.ml_pipeline import MLConfig, MLPipeline, ModelConfig


def cmd_run(args):
    """Run full pipeline."""
    config = _build_config(args)

    print(f"Starting full ML pipeline")
    print(f"Symbol: {config.symbol}")
    print(f"Horizons: {config.horizons}")
    print(f"Models: {[m.name for m in config.models]}")
    print(f"Run ID: {config.run_id}")
    print()

    pipeline = MLPipeline(config)

    try:
        results = pipeline.run()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Run ID: {results['run_id']}")
        print(f"Output: {results['output_dir']}")

        if "data" in results:
            print(f"\nData paths: {results['data'].get('labeled_data_paths', {})}")

        if "training" in results:
            print(f"\nModel paths: {results['training'].get('model_paths', {})}")

        if "evaluation" in results:
            print(f"\nEvaluation complete")

        return 0

    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        return 1


def cmd_data(args):
    """Run data pipeline only."""
    config = _build_config(args)

    print(f"Running data pipeline")
    print(f"Symbol: {config.symbol}")
    print(f"Horizons: {config.horizons}")
    print(f"Run ID: {config.run_id}")
    print()

    pipeline = MLPipeline(config)

    try:
        results = pipeline.run_data()

        print("\n" + "=" * 70)
        print("DATA PIPELINE COMPLETED")
        print("=" * 70)
        print(f"Labeled data paths: {results.get('labeled_data_paths', {})}")
        print(f"Features: {results.get('features_info', {}).get('n_features', 0)} features")

        return 0

    except Exception as e:
        print(f"\nData pipeline failed: {e}", file=sys.stderr)
        return 1


def cmd_train(args):
    """Run training only."""
    config = _build_config(args)

    print(f"Running training")
    print(f"Models: {[m.name for m in config.models]}")
    print(f"Training mode: {config.training_mode}")
    print(f"Run ID: {config.run_id}")
    print()

    pipeline = MLPipeline(config)

    try:
        results = pipeline.run_training()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Model paths: {results.get('model_paths', {})}")
        print(f"Training mode: {results.get('training_mode')}")

        return 0

    except Exception as e:
        print(f"\nTraining failed: {e}", file=sys.stderr)
        return 1


def cmd_evaluate(args):
    """Run evaluation only."""
    config = _build_config(args)

    print(f"Running evaluation")
    print(f"Methods: {config.evaluation_methods}")
    print(f"Run ID: {config.run_id}")
    print()

    pipeline = MLPipeline(config)

    try:
        results = pipeline.run_evaluation()

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETED")
        print("=" * 70)

        for method in config.evaluation_methods:
            if method in results:
                print(f"\n{method}: completed")

        return 0

    except Exception as e:
        print(f"\nEvaluation failed: {e}", file=sys.stderr)
        return 1


def cmd_resume(args):
    """Resume pipeline from checkpoint."""
    if not args.run_id:
        print("Error: --run-id required for resume", file=sys.stderr)
        return 1

    config = _build_config(args)
    config.run_id = args.run_id

    print(f"Resuming pipeline")
    print(f"Run ID: {config.run_id}")
    print()

    pipeline = MLPipeline(config)

    try:
        results = pipeline.resume()

        print("\n" + "=" * 70)
        print("RESUME COMPLETED")
        print("=" * 70)
        print(f"Run ID: {results['run_id']}")

        return 0

    except Exception as e:
        print(f"\nResume failed: {e}", file=sys.stderr)
        return 1


def cmd_status(args):
    """Show pipeline status."""
    if not args.run_id:
        print("Error: --run-id required for status", file=sys.stderr)
        return 1

    from src.ml_pipeline import PipelineState

    try:
        state = PipelineState.load(args.run_id)
        summary = state.get_summary()

        print("=" * 70)
        print("PIPELINE STATUS")
        print("=" * 70)
        print(f"Run ID: {summary['run_id']}")
        print(f"Status: {summary['status']}")
        print(
            f"\nCompleted phases: {', '.join(summary['completed_phases']) if summary['completed_phases'] else 'None'}"
        )
        print(
            f"Pending phases: {', '.join(summary['pending_phases']) if summary['pending_phases'] else 'None'}"
        )

        if summary["start_time"]:
            print(f"\nStart time: {summary['start_time']}")
        if summary["end_time"]:
            print(f"End time: {summary['end_time']}")
        if summary["total_duration_seconds"]:
            print(f"Duration: {summary['total_duration_seconds']:.1f} seconds")

        if summary["error_message"]:
            print(f"\nError: {summary['error_message']}")

        return 0

    except FileNotFoundError:
        print(f"Error: Run ID '{args.run_id}' not found", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _build_config(args) -> MLConfig:
    """Build MLConfig from command-line arguments."""

    if args.config:
        return MLConfig.from_yaml(args.config)

    models = []
    for model_name in args.models:
        models.append(
            ModelConfig(
                name=model_name,
                timeframe=args.timeframe,
                optimize_features=args.optimize_features,
            )
        )

    return MLConfig(
        symbol=args.symbol,
        horizons=args.horizons,
        models=models,
        training_mode=args.training_mode,
        build_ensemble=args.build_ensemble,
        evaluation_methods=args.evaluation_methods,
        start_date=args.start_date,
        end_date=args.end_date,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ml",
        description="Unified ML Pipeline - Train and evaluate ML models on OHLCV data",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser(
        "run", help="Run full pipeline (data + training + evaluation)"
    )
    _add_common_args(run_parser)

    data_parser = subparsers.add_parser("data", help="Run data pipeline only")
    _add_common_args(data_parser)

    train_parser = subparsers.add_parser("train", help="Run training only")
    _add_common_args(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation only")
    _add_common_args(eval_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume from checkpoint")
    resume_parser.add_argument("--run-id", required=True, help="Run ID to resume")
    _add_common_args(resume_parser, include_run_id=False)

    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.add_argument("--run-id", required=True, help="Run ID to check")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "run": cmd_run,
        "data": cmd_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "resume": cmd_resume,
        "status": cmd_status,
    }

    return commands[args.command](args)


def _add_common_args(parser, include_run_id=True):
    """Add common arguments to parser."""
    parser.add_argument("--config", type=Path, help="Path to YAML config file")

    parser.add_argument("--symbol", default="MES", help="Trading symbol (default: MES)")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[20], help="Label horizons (default: 20)"
    )
    parser.add_argument(
        "--models", nargs="+", default=["xgboost"], help="Models to train (default: xgboost)"
    )

    parser.add_argument("--timeframe", default="5min", help="Primary timeframe (default: 5min)")
    parser.add_argument(
        "--training-mode",
        default="standard",
        choices=["standard", "walk_forward", "regime_aware", "meta_labeling"],
        help="Training mode (default: standard)",
    )
    parser.add_argument(
        "--build-ensemble", action="store_true", help="Build ensemble from base models"
    )
    parser.add_argument("--optimize-features", action="store_true", help="Run feature optimization")

    parser.add_argument(
        "--evaluation-methods",
        nargs="+",
        default=[],
        choices=["cv", "walk_forward", "cpcv_pbo"],
        help="Evaluation methods",
    )

    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    if include_run_id:
        parser.add_argument("--run-id", help="Specific run ID (for resume/status)")


if __name__ == "__main__":
    sys.exit(main())
