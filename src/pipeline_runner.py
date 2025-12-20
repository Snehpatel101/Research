"""
Pipeline Runner and Orchestrator
Manages stage execution, dependency tracking, and artifact management

Stage Flow:
1. data_generation    - Generate/validate raw data
2. data_cleaning      - Clean and resample OHLCV data
3. feature_engineering - Generate technical features
4. initial_labeling   - Apply initial triple-barrier labels
5. ga_optimize        - Genetic algorithm optimization of barrier params
6. final_labels       - Apply optimized labels with quality scores
7. create_splits      - Create train/val/test splits
8. validate           - Comprehensive data validation
9. generate_report    - Generate completion report
"""
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Callable, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback
import pandas as pd
import numpy as np

from pipeline_config import PipelineConfig
from manifest import ArtifactManifest


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of executing a pipeline stage."""
    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    artifacts: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'stage_name': self.stage_name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'artifacts': [str(p) for p in self.artifacts],
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass
class PipelineStage:
    """Definition of a pipeline stage."""
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    required: bool = True
    can_run_parallel: bool = False


class PipelineRunner:
    """Orchestrates the Phase 1 pipeline execution."""

    def __init__(self, config: PipelineConfig, resume: bool = False):
        """
        Initialize pipeline runner.

        Args:
            config: Pipeline configuration
            resume: Whether to resume from last successful stage
        """
        self.config = config
        self.resume = resume
        self.manifest = ArtifactManifest(config.run_id, config.project_root)

        # Set up logging
        self.config.create_directories()
        self.log_file = self.config.run_logs_dir / "pipeline.log"
        self.setup_logging()

        self.logger = logging.getLogger(__name__)

        # Stage tracking
        self.stages: List[PipelineStage] = []
        self.stage_results: Dict[str, StageResult] = {}
        self.completed_stages: Set[str] = set()

        # Define pipeline stages
        self._define_stages()

        # Load previous state if resuming
        if self.resume:
            self._load_state()

    def setup_logging(self):
        """Configure logging for the pipeline."""
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def _define_stages(self):
        """Define all pipeline stages and their dependencies."""
        self.stages = [
            PipelineStage(
                name="data_generation",
                function=self._run_data_generation,
                dependencies=[],
                description="Stage 1: Generate or validate raw data files",
                required=True
            ),
            PipelineStage(
                name="data_cleaning",
                function=self._run_data_cleaning,
                dependencies=["data_generation"],
                description="Stage 2: Clean and resample OHLCV data",
                required=True
            ),
            PipelineStage(
                name="feature_engineering",
                function=self._run_feature_engineering,
                dependencies=["data_cleaning"],
                description="Stage 3: Generate technical features",
                required=True
            ),
            PipelineStage(
                name="initial_labeling",
                function=self._run_initial_labeling,
                dependencies=["feature_engineering"],
                description="Stage 4: Apply initial triple-barrier labeling",
                required=True
            ),
            PipelineStage(
                name="ga_optimize",
                function=self._run_ga_optimization,
                dependencies=["initial_labeling"],
                description="Stage 5: GA optimization of barrier parameters",
                required=True
            ),
            PipelineStage(
                name="final_labels",
                function=self._run_final_labels,
                dependencies=["ga_optimize"],
                description="Stage 6: Apply optimized labels with quality scores",
                required=True
            ),
            PipelineStage(
                name="create_splits",
                function=self._run_create_splits,
                dependencies=["final_labels"],
                description="Stage 7: Create train/val/test splits",
                required=True
            ),
            PipelineStage(
                name="validate",
                function=self._run_validation,
                dependencies=["create_splits"],
                description="Stage 8: Comprehensive data validation",
                required=True
            ),
            PipelineStage(
                name="generate_report",
                function=self._run_generate_report,
                dependencies=["validate"],
                description="Stage 9: Generate completion report",
                required=True
            )
        ]

    def _run_data_generation(self) -> StageResult:
        """Stage 1: Data Generation and Ingestion with Validation."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 1: Data Generation / Acquisition & Validation")
        self.logger.info("="*70)

        try:
            from generate_synthetic_data import main as generate_data
            from stages.stage1_ingest import DataIngestor

            # Check if raw data exists
            raw_files_exist = all(
                (self.config.raw_data_dir / f"{s}_1m.parquet").exists() or
                (self.config.raw_data_dir / f"{s}_1m.csv").exists()
                for s in self.config.symbols
            )

            artifacts = []
            ingestion_metadata = {}

            # Step 1: Generate synthetic data if needed
            if not raw_files_exist or self.config.use_synthetic_data:
                self.logger.info("Generating synthetic data...")
                generate_data()
                self.logger.info("Synthetic data generation complete.")
            else:
                self.logger.info("Raw data files already exist. Skipping generation.")

            # Step 2: ALWAYS run DataIngestor for validation and standardization
            self.logger.info("\nRunning DataIngestor for validation and standardization...")

            # Create validated data output directory
            validated_data_dir = self.config.raw_data_dir / "validated"
            validated_data_dir.mkdir(parents=True, exist_ok=True)

            # Initialize DataIngestor
            ingestor = DataIngestor(
                raw_data_dir=self.config.raw_data_dir,
                output_dir=validated_data_dir,
                source_timezone='UTC',
                symbol_col='symbol'
            )

            # Process each symbol
            total_violations = 0
            for symbol in self.config.symbols:
                # Find the raw data file
                raw_file = None
                for ext in ['.parquet', '.csv']:
                    candidate = self.config.raw_data_dir / f"{symbol}_1m{ext}"
                    if candidate.exists():
                        raw_file = candidate
                        break

                if raw_file is None:
                    raise FileNotFoundError(
                        f"Raw data file not found for symbol {symbol}. "
                        f"Expected: {self.config.raw_data_dir}/{symbol}_1m.parquet or .csv"
                    )

                self.logger.info(f"\nProcessing {symbol} from {raw_file.name}...")

                # Ingest and validate the file
                df, metadata = ingestor.ingest_file(
                    file_path=raw_file,
                    symbol=symbol,
                    validate=True
                )

                # Check for validation issues
                validation_info = metadata.get('validation', {})
                violations = validation_info.get('violations', {})
                if violations:
                    violation_count = sum(violations.values())
                    total_violations += violation_count
                    self.logger.warning(
                        f"Symbol {symbol}: Fixed {violation_count} OHLCV violations: {violations}"
                    )

                # Save validated data
                output_path = ingestor.save_parquet(df, f"{symbol}_1m_validated", metadata)
                artifacts.append(output_path)

                # Store metadata for this symbol
                ingestion_metadata[symbol] = {
                    'source_file': str(raw_file),
                    'validated_file': str(output_path),
                    'raw_rows': metadata.get('raw_rows', 0),
                    'final_rows': metadata.get('final_rows', 0),
                    'date_range': metadata.get('date_range', {}),
                    'violations_fixed': sum(violations.values()) if violations else 0,
                    'validation_details': violations
                }

                # Add to manifest
                self.manifest.add_artifact(
                    name=f"validated_data_{symbol}",
                    file_path=output_path,
                    stage="data_generation",
                    metadata=ingestion_metadata[symbol]
                )

                self.logger.info(
                    f"Validated {symbol}: {metadata.get('raw_rows', 0):,} -> "
                    f"{metadata.get('final_rows', 0):,} rows"
                )

            # Log summary
            self.logger.info("\n" + "-"*50)
            self.logger.info("INGESTION SUMMARY")
            self.logger.info("-"*50)
            for symbol, meta in ingestion_metadata.items():
                self.logger.info(
                    f"  {symbol}: {meta['raw_rows']:,} raw -> {meta['final_rows']:,} validated "
                    f"({meta['violations_fixed']} fixes)"
                )
            if total_violations > 0:
                self.logger.warning(f"Total OHLCV violations fixed: {total_violations}")
            else:
                self.logger.info("No OHLCV violations found - data is clean!")

            end_time = datetime.now()
            return StageResult(
                stage_name="data_generation",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata={
                    'symbols': self.config.symbols,
                    'ingestion_results': ingestion_metadata,
                    'total_violations_fixed': total_violations
                }
            )

        except Exception as e:
            self.logger.error(f"Data generation/ingestion failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="data_generation",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_data_cleaning(self) -> StageResult:
        """Stage 2: Data Cleaning - uses validated data from Stage 1."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 2: Data Cleaning")
        self.logger.info("="*70)

        try:
            from data_cleaning import clean_symbol_data

            # Use validated data from Stage 1
            validated_data_dir = self.config.raw_data_dir / "validated"

            artifacts = []
            for symbol in self.config.symbols:
                # Look for validated data first, fall back to raw if not found
                input_path = validated_data_dir / f"{symbol}_1m_validated.parquet"
                if not input_path.exists():
                    # Fall back to raw data (for backward compatibility)
                    self.logger.warning(
                        f"Validated data not found for {symbol}, using raw data"
                    )
                    input_path = self.config.raw_data_dir / f"{symbol}_1m.parquet"
                    if not input_path.exists():
                        input_path = self.config.raw_data_dir / f"{symbol}_1m.csv"

                if not input_path.exists():
                    raise FileNotFoundError(f"No input data found for {symbol}")

                output_path = self.config.clean_data_dir / f"{symbol}_5m_clean.parquet"

                self.logger.info(f"Cleaning {symbol}: {input_path.name} -> {output_path.name}")
                clean_symbol_data(input_path, output_path, symbol)

                if output_path.exists():
                    artifacts.append(output_path)
                    self.manifest.add_artifact(
                        name=f"clean_data_{symbol}",
                        file_path=output_path,
                        stage="data_cleaning",
                        metadata={'symbol': symbol, 'source': str(input_path)}
                    )

            end_time = datetime.now()
            return StageResult(
                stage_name="data_cleaning",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts
            )

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="data_cleaning",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_feature_engineering(self) -> StageResult:
        """Stage 3: Feature Engineering."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 3: Feature Engineering")
        self.logger.info("="*70)

        try:
            from feature_engineering import main as generate_features
            generate_features()

            artifacts = []
            for symbol in self.config.symbols:
                file_path = self.config.features_dir / f"{symbol}_5m_features.parquet"
                if file_path.exists():
                    artifacts.append(file_path)
                    self.manifest.add_artifact(
                        name=f"features_{symbol}",
                        file_path=file_path,
                        stage="feature_engineering",
                        metadata={'symbol': symbol}
                    )

            end_time = datetime.now()
            return StageResult(
                stage_name="feature_engineering",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts
            )

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="feature_engineering",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_initial_labeling(self) -> StageResult:
        """Stage 4: Initial Triple-Barrier Labeling for GA optimization input."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 4: Initial Triple-Barrier Labeling")
        self.logger.info("="*70)

        try:
            from stages.stage4_labeling import triple_barrier_numba

            # Create labels directory for GA input
            labels_dir = self.config.project_root / 'data' / 'labels'
            labels_dir.mkdir(parents=True, exist_ok=True)

            artifacts = []
            label_stats = {}

            for symbol in self.config.symbols:
                # Load features data
                features_path = self.config.features_dir / f"{symbol}_5m_features.parquet"
                if not features_path.exists():
                    raise FileNotFoundError(f"Features file not found: {features_path}")

                self.logger.info(f"\nProcessing {symbol}...")
                df = pd.read_parquet(features_path)
                self.logger.info(f"  Loaded {len(df):,} rows")

                # Check for ATR column
                atr_col = 'atr_14'
                if atr_col not in df.columns:
                    raise ValueError(f"ATR column '{atr_col}' not found in features")

                # Apply initial labeling with default parameters for each horizon
                for horizon in self.config.label_horizons:
                    # Default parameters (will be optimized by GA)
                    k_up = 2.0
                    k_down = 1.0
                    max_bars = horizon * 3

                    self.logger.info(f"  Horizon {horizon}: k_up={k_up}, k_down={k_down}, max_bars={max_bars}")

                    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
                        df['close'].values,
                        df['high'].values,
                        df['low'].values,
                        df['open'].values,
                        df[atr_col].values,
                        k_up, k_down, max_bars
                    )

                    # Add columns
                    df[f'label_h{horizon}'] = labels
                    df[f'bars_to_hit_h{horizon}'] = bars_to_hit
                    df[f'mae_h{horizon}'] = mae
                    df[f'mfe_h{horizon}'] = mfe

                    # Log distribution
                    n_long = (labels == 1).sum()
                    n_short = (labels == -1).sum()
                    n_neutral = (labels == 0).sum()
                    total = len(labels)
                    self.logger.info(f"    Distribution: L={n_long/total*100:.1f}% S={n_short/total*100:.1f}% N={n_neutral/total*100:.1f}%")

                # Save to labels directory for GA input
                output_path = labels_dir / f"{symbol}_labels_init.parquet"
                df.to_parquet(output_path, index=False)
                artifacts.append(output_path)

                self.manifest.add_artifact(
                    name=f"initial_labels_{symbol}",
                    file_path=output_path,
                    stage="initial_labeling",
                    metadata={'symbol': symbol, 'horizons': self.config.label_horizons}
                )

                self.logger.info(f"  Saved initial labels to {output_path}")

            end_time = datetime.now()
            return StageResult(
                stage_name="initial_labeling",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata={'horizons': self.config.label_horizons}
            )

        except Exception as e:
            self.logger.error(f"Initial labeling failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="initial_labeling",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_ga_optimization(self) -> StageResult:
        """Stage 5: Genetic Algorithm Optimization of labeling parameters."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 5: Genetic Algorithm Optimization")
        self.logger.info("="*70)

        try:
            from stages.stage5_ga_optimize import run_ga_optimization, plot_convergence

            # GA results directory
            ga_results_dir = self.config.project_root / 'config' / 'ga_results'
            ga_results_dir.mkdir(parents=True, exist_ok=True)

            # Plots directory
            plots_dir = self.config.results_dir / 'ga_plots'
            plots_dir.mkdir(parents=True, exist_ok=True)

            artifacts = []
            all_results = {}

            # GA configuration
            population_size = 50
            generations = 30

            self.logger.info(f"GA Config: population={population_size}, generations={generations}")
            self.logger.info(f"Horizons: {self.config.label_horizons}")

            for symbol in self.config.symbols:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Optimizing {symbol}")
                self.logger.info(f"{'='*50}")

                # Load labels data
                labels_dir = self.config.project_root / 'data' / 'labels'
                labels_path = labels_dir / f"{symbol}_labels_init.parquet"

                if not labels_path.exists():
                    raise FileNotFoundError(f"Labels file not found: {labels_path}")

                df = pd.read_parquet(labels_path)
                self.logger.info(f"Loaded {len(df):,} rows")

                symbol_results = {}

                for horizon in self.config.label_horizons:
                    # Check if already optimized (skip if results exist)
                    results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"

                    if results_path.exists():
                        self.logger.info(f"\n  Horizon {horizon}: Loading existing results from {results_path.name}")
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        symbol_results[horizon] = results
                        artifacts.append(results_path)
                        self.logger.info(f"    k_up={results['best_k_up']:.3f}, k_down={results['best_k_down']:.3f}, "
                                        f"max_bars={results['best_max_bars']}, fitness={results['best_fitness']:.4f}")
                        continue

                    # Run GA optimization
                    self.logger.info(f"\n  Horizon {horizon}: Running GA optimization...")

                    results, logbook = run_ga_optimization(
                        df, horizon,
                        population_size=population_size,
                        generations=generations,
                        subset_fraction=0.3,
                        atr_column='atr_14'
                    )

                    symbol_results[horizon] = results

                    # Save results
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    artifacts.append(results_path)

                    self.logger.info(f"    Best: k_up={results['best_k_up']:.3f}, k_down={results['best_k_down']:.3f}, "
                                    f"max_bars={results['best_max_bars']}, fitness={results['best_fitness']:.4f}")

                    # Check signal rate
                    val = results.get('validation', {})
                    signal_rate = val.get('signal_rate', 0)
                    if signal_rate < 0.40:
                        self.logger.warning(f"    WARNING: Signal rate {signal_rate*100:.1f}% below 40% threshold!")

                    # Plot convergence
                    plot_path = plots_dir / f"{symbol}_ga_h{horizon}_convergence.png"
                    plot_convergence(results, plot_path)
                    artifacts.append(plot_path)

                    self.manifest.add_artifact(
                        name=f"ga_results_{symbol}_h{horizon}",
                        file_path=results_path,
                        stage="ga_optimize",
                        metadata={
                            'symbol': symbol,
                            'horizon': horizon,
                            'best_fitness': results['best_fitness'],
                            'signal_rate': signal_rate
                        }
                    )

                all_results[symbol] = symbol_results

            # Save combined summary
            summary = {}
            for symbol, symbol_results in all_results.items():
                summary[symbol] = {
                    str(h): {
                        'k_up': res['best_k_up'],
                        'k_down': res['best_k_down'],
                        'max_bars': res['best_max_bars'],
                        'fitness': res['best_fitness'],
                        'signal_rate': res.get('validation', {}).get('signal_rate', None)
                    }
                    for h, res in symbol_results.items()
                }

            summary_path = ga_results_dir / 'optimization_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            artifacts.append(summary_path)

            self.logger.info(f"\nOptimization summary saved to {summary_path}")

            end_time = datetime.now()
            return StageResult(
                stage_name="ga_optimize",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata={'all_results': summary}
            )

        except Exception as e:
            self.logger.error(f"GA optimization failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="ga_optimize",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_final_labels(self) -> StageResult:
        """Stage 6: Apply optimized labels with quality scores and sample weights."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 6: Final Labels with Quality Scores")
        self.logger.info("="*70)

        try:
            from stages.stage6_final_labels import apply_optimized_labels, generate_labeling_report

            # GA results directory
            ga_results_dir = self.config.project_root / 'config' / 'ga_results'

            artifacts = []
            all_dfs = {}

            for symbol in self.config.symbols:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Processing {symbol}")
                self.logger.info(f"{'='*50}")

                # Load features data (original, without initial labels)
                features_path = self.config.features_dir / f"{symbol}_5m_features.parquet"
                if not features_path.exists():
                    raise FileNotFoundError(f"Features file not found: {features_path}")

                df = pd.read_parquet(features_path)
                self.logger.info(f"Loaded {len(df):,} rows from features")

                # Apply optimized labels for each horizon
                for horizon in self.config.label_horizons:
                    results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"

                    if results_path.exists():
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        best_params = {
                            'k_up': results['best_k_up'],
                            'k_down': results['best_k_down'],
                            'max_bars': results['best_max_bars']
                        }
                        self.logger.info(f"\n  Horizon {horizon}: Using GA-optimized params")
                    else:
                        # Fall back to defaults if no GA results
                        self.logger.warning(f"  Horizon {horizon}: No GA results found, using defaults")
                        best_params = {
                            'k_up': 2.0,
                            'k_down': 1.0,
                            'max_bars': horizon * 3
                        }

                    # Apply optimized labeling with quality scores
                    df = apply_optimized_labels(df, horizon, best_params, atr_column='atr_14')

                # Save final labeled data
                output_path = self.config.final_data_dir / f"{symbol}_labeled.parquet"
                df.to_parquet(output_path, index=False)
                artifacts.append(output_path)

                self.manifest.add_artifact(
                    name=f"final_labeled_{symbol}",
                    file_path=output_path,
                    stage="final_labels",
                    metadata={'symbol': symbol, 'horizons': self.config.label_horizons}
                )

                all_dfs[symbol] = df
                self.logger.info(f"\n  Saved final labels to {output_path}")

            # Generate labeling report
            if all_dfs:
                generate_labeling_report(all_dfs)
                report_path = self.config.results_dir / 'labeling_report.md'
                if report_path.exists():
                    artifacts.append(report_path)

            end_time = datetime.now()
            return StageResult(
                stage_name="final_labels",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata={'symbols': self.config.symbols, 'horizons': self.config.label_horizons}
            )

        except Exception as e:
            self.logger.error(f"Final labeling failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="final_labels",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_create_splits(self) -> StageResult:
        """Stage 7: Create Train/Val/Test Splits."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 7: Create Train/Val/Test Splits")
        self.logger.info("="*70)

        try:
            # Load and combine labeled data
            dfs = []
            for symbol in self.config.symbols:
                fpath = self.config.final_data_dir / f"{symbol}_labeled.parquet"
                if fpath.exists():
                    df = pd.read_parquet(fpath)
                    dfs.append(df)
                    self.logger.info(f"Loaded {len(df):,} rows for {symbol}")

            if not dfs:
                raise RuntimeError("No labeled data found!")

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            self.logger.info(f"Combined dataset: {len(combined_df):,} rows")

            # Save combined dataset
            combined_path = self.config.final_data_dir / "combined_final_labeled.parquet"
            combined_df.to_parquet(combined_path, index=False)
            self.logger.info(f"Saved combined dataset to {combined_path}")

            # Create splits
            n = len(combined_df)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

            # Apply purging
            train_end_purged = train_end - self.config.purge_bars
            val_start = train_end + self.config.embargo_bars
            test_start = val_end + self.config.embargo_bars

            # Create indices
            train_indices = np.arange(0, train_end_purged)
            val_indices = np.arange(val_start, val_end)
            test_indices = np.arange(test_start, n)

            self.logger.info(f"Split sizes:")
            self.logger.info(f"  Train: {len(train_indices):,} samples")
            self.logger.info(f"  Val:   {len(val_indices):,} samples")
            self.logger.info(f"  Test:  {len(test_indices):,} samples")

            # Get date ranges
            train_dates = combined_df.iloc[train_indices]['datetime']
            val_dates = combined_df.iloc[val_indices]['datetime']
            test_dates = combined_df.iloc[test_indices]['datetime']

            self.logger.info(f"Date ranges:")
            self.logger.info(f"  Train: {train_dates.min()} to {train_dates.max()}")
            self.logger.info(f"  Val:   {val_dates.min()} to {val_dates.max()}")
            self.logger.info(f"  Test:  {test_dates.min()} to {test_dates.max()}")

            # Save indices
            self.config.splits_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.config.splits_dir / "train_indices.npy", train_indices)
            np.save(self.config.splits_dir / "val_indices.npy", val_indices)
            np.save(self.config.splits_dir / "test_indices.npy", test_indices)

            # Save metadata
            split_config = {
                "run_id": self.config.run_id,
                "total_samples": n,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "purge_bars": self.config.purge_bars,
                "embargo_bars": self.config.embargo_bars,
                "train_date_start": str(train_dates.min()),
                "train_date_end": str(train_dates.max()),
                "val_date_start": str(val_dates.min()),
                "val_date_end": str(val_dates.max()),
                "test_date_start": str(test_dates.min()),
                "test_date_end": str(test_dates.max()),
            }

            with open(self.config.splits_dir / "split_config.json", 'w') as f:
                json.dump(split_config, f, indent=2)

            self.logger.info(f"Saved splits to {self.config.splits_dir}")

            # Track artifacts
            artifacts = [
                combined_path,
                self.config.splits_dir / "train_indices.npy",
                self.config.splits_dir / "val_indices.npy",
                self.config.splits_dir / "test_indices.npy",
                self.config.splits_dir / "split_config.json"
            ]

            for artifact_path in artifacts:
                self.manifest.add_artifact(
                    name=f"splits_{artifact_path.name}",
                    file_path=artifact_path,
                    stage="create_splits",
                    metadata=split_config
                )

            end_time = datetime.now()
            return StageResult(
                stage_name="create_splits",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata=split_config
            )

        except Exception as e:
            self.logger.error(f"Create splits failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="create_splits",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_validation(self) -> StageResult:
        """Stage 8: Comprehensive data validation with optional pipeline failure."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 8: Data Validation")
        self.logger.info("="*70)

        try:
            from stages.stage8_validate import validate_data, DataValidator

            # Path to combined labeled data
            combined_path = self.config.final_data_dir / "combined_final_labeled.parquet"

            if not combined_path.exists():
                raise FileNotFoundError(f"Combined labeled data not found: {combined_path}")

            # Output paths
            validation_report_path = self.config.results_dir / f"validation_report_{self.config.run_id}.json"
            feature_selection_path = self.config.results_dir / f"feature_selection_{self.config.run_id}.json"

            self.logger.info(f"Validating combined dataset: {combined_path}")

            # Run validation with feature selection
            summary, feature_selection_result = validate_data(
                data_path=combined_path,
                output_path=validation_report_path,
                horizons=self.config.label_horizons,
                run_feature_selection=True,
                correlation_threshold=0.85,
                variance_threshold=0.01,
                feature_selection_output_path=feature_selection_path
            )

            artifacts = [validation_report_path]
            if feature_selection_path.exists():
                artifacts.append(feature_selection_path)

            # Log summary
            self.logger.info(f"\n{'='*50}")
            self.logger.info("VALIDATION SUMMARY")
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Status: {summary['status']}")
            self.logger.info(f"Issues: {summary['issues_count']}")
            self.logger.info(f"Warnings: {summary['warnings_count']}")

            if summary['issues']:
                self.logger.error("Critical Issues Found:")
                for issue in summary['issues'][:10]:  # Show first 10
                    self.logger.error(f"  - {issue}")
                if len(summary['issues']) > 10:
                    self.logger.error(f"  ... and {len(summary['issues'])-10} more")

            if summary['warnings']:
                self.logger.warning("Warnings:")
                for warning in summary['warnings'][:10]:
                    self.logger.warning(f"  - {warning}")

            # Feature selection results
            if feature_selection_result:
                self.logger.info(f"\nFeature Selection:")
                self.logger.info(f"  Original features: {feature_selection_result.original_count}")
                self.logger.info(f"  Selected features: {feature_selection_result.final_count}")
                self.logger.info(f"  Removed: {len(feature_selection_result.removed_features)}")

            # Add to manifest
            self.manifest.add_artifact(
                name="validation_report",
                file_path=validation_report_path,
                stage="validate",
                metadata={
                    'status': summary['status'],
                    'issues_count': summary['issues_count'],
                    'warnings_count': summary['warnings_count']
                }
            )

            end_time = datetime.now()

            # Determine if validation passed or failed
            if summary['status'] == 'FAILED':
                self.logger.error(f"\nValidation FAILED with {summary['issues_count']} critical issues")
                return StageResult(
                    stage_name="validate",
                    status=StageStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    artifacts=artifacts,
                    error=f"Validation failed with {summary['issues_count']} critical issues",
                    metadata=summary
                )

            self.logger.info(f"\nValidation PASSED (with {summary['warnings_count']} warnings)")
            return StageResult(
                stage_name="validate",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata=summary
            )

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="validate",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_generate_report(self) -> StageResult:
        """Stage 9: Generate Completion Report."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 9: Generate Completion Report")
        self.logger.info("="*70)

        try:
            # Load combined data
            combined_path = self.config.final_data_dir / "combined_final_labeled.parquet"
            combined_df = pd.read_parquet(combined_path)

            # Load split config
            with open(self.config.splits_dir / "split_config.json", 'r') as f:
                split_config = json.load(f)

            # Count features
            feature_cols = [c for c in combined_df.columns
                            if c not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                            and not c.startswith('label_') and not c.startswith('bars_to_hit_')
                            and not c.startswith('mae_') and not c.startswith('quality_')
                            and not c.startswith('sample_weight_')]

            # Label statistics
            label_stats = {}
            for horizon in self.config.label_horizons:
                col = f'label_h{horizon}'
                if col in combined_df.columns:
                    counts = combined_df[col].value_counts().sort_index()
                    label_stats[horizon] = {
                        'short': int(counts.get(-1, 0)),
                        'neutral': int(counts.get(0, 0)),
                        'long': int(counts.get(1, 0))
                    }

            # Generate report content
            report = self._generate_report_content(
                combined_df, split_config, feature_cols, label_stats
            )

            # Save report
            report_path = self.config.results_dir / f"PHASE1_COMPLETION_REPORT_{self.config.run_id}.md"
            with open(report_path, 'w') as f:
                f.write(report)

            self.logger.info(f"Report saved to: {report_path}")

            # Add to manifest
            self.manifest.add_artifact(
                name="completion_report",
                file_path=report_path,
                stage="generate_report",
                metadata={
                    'total_samples': len(combined_df),
                    'num_features': len(feature_cols)
                }
            )

            end_time = datetime.now()
            return StageResult(
                stage_name="generate_report",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=[report_path]
            )

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="generate_report",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _generate_report_content(
        self,
        combined_df: pd.DataFrame,
        split_config: Dict,
        feature_cols: List[str],
        label_stats: Dict
    ) -> str:
        """Generate the report markdown content."""
        report = f'''# Phase 1 Completion Report
## Ensemble Price Prediction System

**Run ID:** {self.config.run_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

Phase 1 successfully processed raw OHLCV data through the complete pipeline:
- Data Cleaning (1-min to 5-min resampling)
- Feature Engineering ({len(feature_cols)} technical features)
- Triple-Barrier Labeling ({len(self.config.label_horizons)} horizons: {', '.join(map(str, self.config.label_horizons))} bars)
- Train/Val/Test Splits (with purging & embargo)

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | {len(combined_df):,} |
| **Symbols** | {', '.join(self.config.symbols)} |
| **Date Range** | {combined_df['datetime'].min()} to {combined_df['datetime'].max()} |
| **Resolution** | {self.config.bar_resolution} |
| **Features** | {len(feature_cols)} |

---

## Label Distribution

'''
        # Add label distribution for each horizon
        for horizon in self.config.label_horizons:
            stats = label_stats.get(horizon, {})
            total = sum(stats.values()) if stats else len(combined_df)
            report += f'''### Horizon {horizon} ({horizon}-bar ahead)
| Class | Count | Percentage |
|-------|-------|------------|
| Short (-1) | {stats.get('short', 0):,} | {stats.get('short', 0)/total*100:.1f}% |
| Neutral (0) | {stats.get('neutral', 0):,} | {stats.get('neutral', 0)/total*100:.1f}% |
| Long (+1) | {stats.get('long', 0):,} | {stats.get('long', 0)/total*100:.1f}% |

'''

        report += f'''---

## Data Splits

| Split | Samples | Percentage | Date Range |
|-------|---------|------------|------------|
| **Train** | {split_config['train_samples']:,} | {split_config['train_samples']/split_config['total_samples']*100:.1f}% | {split_config['train_date_start'][:10]} to {split_config['train_date_end'][:10]} |
| **Validation** | {split_config['val_samples']:,} | {split_config['val_samples']/split_config['total_samples']*100:.1f}% | {split_config['val_date_start'][:10]} to {split_config['val_date_end'][:10]} |
| **Test** | {split_config['test_samples']:,} | {split_config['test_samples']/split_config['total_samples']*100:.1f}% | {split_config['test_date_start'][:10]} to {split_config['test_date_end'][:10]} |

### Leakage Prevention
- **Purge bars:** {split_config['purge_bars']} bars removed at boundaries
- **Embargo period:** {split_config['embargo_bars']} bars buffer

---

## Pipeline Execution Summary

'''
        # Add stage execution summary
        for stage_name, result in self.stage_results.items():
            status_emoji = "✅" if result.status == StageStatus.COMPLETED else "❌"
            report += f'''### {status_emoji} {stage_name.replace('_', ' ').title()}
- **Status:** {result.status.value}
- **Duration:** {result.duration_seconds:.2f} seconds
- **Artifacts:** {len(result.artifacts)}

'''

        report += f'''---

## Next Steps: Phase 2

1. Load training data with splits
2. Train base models (N-HiTS, TFT, PatchTST)
3. Use sample weights for quality-aware training

```python
import numpy as np
import pandas as pd

# Load data and splits
train_idx = np.load('data/splits/train_indices.npy')
df = pd.read_parquet('data/final/combined_final_labeled.parquet')
train_df = df.iloc[train_idx]

# Get features and labels
X_train = train_df[{feature_cols[:5]}].values  # Example features
y_train = train_df['label_h5'].values  # For 5-bar horizon
sample_weights = train_df['sample_weight_h5'].values
```

---

*Phase 1 Complete - Ready for Phase 2*
'''
        return report

    def run(self, from_stage: Optional[str] = None) -> bool:
        """
        Run the complete pipeline.

        Args:
            from_stage: Stage name to resume from (None to run all)

        Returns:
            True if all stages completed successfully
        """
        pipeline_start = datetime.now()

        self.logger.info("="*70)
        self.logger.info("PHASE 1: DATA PREPARATION PIPELINE")
        self.logger.info(f"Run ID: {self.config.run_id}")
        self.logger.info("="*70)

        # Save configuration
        self.config.save_config()
        self.logger.info(f"Configuration saved to {self.config.run_config_dir / 'config.json'}")

        # Determine which stages to run
        stages_to_run = self._get_stages_to_run(from_stage)

        # Execute stages
        all_success = True
        for stage in stages_to_run:
            # Check dependencies
            if not self._check_dependencies(stage):
                self.logger.error(f"Dependencies not met for stage: {stage.name}")
                all_success = False
                break

            # Execute stage
            self.logger.info(f"\nExecuting stage: {stage.name}")
            result = stage.function()

            # Store result
            self.stage_results[stage.name] = result

            if result.status == StageStatus.COMPLETED:
                self.completed_stages.add(stage.name)
                self.logger.info(f"✅ Stage completed: {stage.name} ({result.duration_seconds:.2f}s)")
            else:
                self.logger.error(f"❌ Stage failed: {stage.name}")
                if result.error:
                    self.logger.error(f"Error: {result.error}")
                all_success = False
                if stage.required:
                    self.logger.error("Required stage failed. Stopping pipeline.")
                    break

            # Save state after each stage
            self._save_state()

        # Save final manifest
        self.manifest.save()
        self.logger.info(f"\nManifest saved to {self.manifest.manifest_path}")

        # Final summary
        pipeline_end = datetime.now()
        total_duration = (pipeline_end - pipeline_start).total_seconds()

        self.logger.info("\n" + "="*70)
        if all_success:
            self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        else:
            self.logger.info("❌ PIPELINE FAILED")
        self.logger.info("="*70)
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        self.logger.info(f"Completed stages: {len(self.completed_stages)}/{len(self.stages)}")
        self.logger.info(f"Run ID: {self.config.run_id}")
        self.logger.info(f"Logs: {self.log_file}")

        return all_success

    def _get_stages_to_run(self, from_stage: Optional[str]) -> List[PipelineStage]:
        """Determine which stages to run based on resume point."""
        if from_stage is None:
            return self.stages

        # Find the index of the from_stage
        start_idx = None
        for idx, stage in enumerate(self.stages):
            if stage.name == from_stage:
                start_idx = idx
                break

        if start_idx is None:
            raise ValueError(f"Stage not found: {from_stage}")

        return self.stages[start_idx:]

    def _check_dependencies(self, stage: PipelineStage) -> bool:
        """Check if all dependencies for a stage are completed."""
        for dep in stage.dependencies:
            if dep not in self.completed_stages:
                return False
        return True

    def _save_state(self):
        """Save current pipeline state."""
        state = {
            'run_id': self.config.run_id,
            'completed_stages': list(self.completed_stages),
            'stage_results': {
                name: result.to_dict()
                for name, result in self.stage_results.items()
            },
            'saved_at': datetime.now().isoformat()
        }

        state_path = self.config.run_artifacts_dir / "pipeline_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load previous pipeline state for resuming."""
        state_path = self.config.run_artifacts_dir / "pipeline_state.json"

        if not state_path.exists():
            self.logger.warning("No previous state found. Starting from beginning.")
            return

        with open(state_path, 'r') as f:
            state = json.load(f)

        self.completed_stages = set(state.get('completed_stages', []))
        self.logger.info(f"Loaded state with {len(self.completed_stages)} completed stages")

        # Load manifest if exists
        try:
            self.manifest = ArtifactManifest.load(
                self.config.run_id,
                self.config.project_root
            )
        except FileNotFoundError:
            self.logger.warning("No previous manifest found.")


if __name__ == "__main__":
    # Example usage
    from pipeline_config import create_default_config

    config = create_default_config(
        symbols=['MES', 'MGC'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        description='Test pipeline run'
    )

    runner = PipelineRunner(config)
    success = runner.run()

    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
