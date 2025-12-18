"""
Pipeline Runner and Orchestrator
Manages stage execution, dependency tracking, and artifact management
"""
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Callable, Set, Any
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
                description="Generate or validate raw data files",
                required=True
            ),
            PipelineStage(
                name="data_cleaning",
                function=self._run_data_cleaning,
                dependencies=["data_generation"],
                description="Clean and resample OHLCV data",
                required=True
            ),
            PipelineStage(
                name="feature_engineering",
                function=self._run_feature_engineering,
                dependencies=["data_cleaning"],
                description="Generate technical features",
                required=True
            ),
            PipelineStage(
                name="labeling",
                function=self._run_labeling,
                dependencies=["feature_engineering"],
                description="Apply triple-barrier labeling",
                required=True
            ),
            PipelineStage(
                name="create_splits",
                function=self._run_create_splits,
                dependencies=["labeling"],
                description="Create train/val/test splits",
                required=True
            ),
            PipelineStage(
                name="generate_report",
                function=self._run_generate_report,
                dependencies=["create_splits"],
                description="Generate completion report",
                required=True
            )
        ]

    def _run_data_generation(self) -> StageResult:
        """Stage 1: Data Generation or Validation."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 1: Data Generation / Acquisition")
        self.logger.info("="*70)

        try:
            from generate_synthetic_data import main as generate_data

            # Check if raw data exists
            raw_files_exist = all(
                (self.config.raw_data_dir / f"{s}_1m.parquet").exists() or
                (self.config.raw_data_dir / f"{s}_1m.csv").exists()
                for s in self.config.symbols
            )

            artifacts = []

            if not raw_files_exist or self.config.use_synthetic_data:
                self.logger.info("Generating synthetic data...")
                generate_data()

                for symbol in self.config.symbols:
                    file_path = self.config.raw_data_dir / f"{symbol}_1m.parquet"
                    if file_path.exists():
                        artifacts.append(file_path)
                        self.manifest.add_artifact(
                            name=f"raw_data_{symbol}",
                            file_path=file_path,
                            stage="data_generation",
                            metadata={'symbol': symbol}
                        )
            else:
                self.logger.info("Raw data files already exist. Skipping generation.")
                for symbol in self.config.symbols:
                    for ext in ['.parquet', '.csv']:
                        file_path = self.config.raw_data_dir / f"{symbol}_1m{ext}"
                        if file_path.exists():
                            artifacts.append(file_path)
                            self.manifest.add_artifact(
                                name=f"raw_data_{symbol}",
                                file_path=file_path,
                                stage="data_generation",
                                metadata={'symbol': symbol}
                            )
                            break

            end_time = datetime.now()
            return StageResult(
                stage_name="data_generation",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                metadata={'symbols': self.config.symbols}
            )

        except Exception as e:
            self.logger.error(f"Data generation failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="data_generation",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_data_cleaning(self) -> StageResult:
        """Stage 2: Data Cleaning."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 2: Data Cleaning")
        self.logger.info("="*70)

        try:
            from data_cleaning import main as clean_data
            clean_data()

            artifacts = []
            for symbol in self.config.symbols:
                file_path = self.config.clean_data_dir / f"{symbol}_5m_clean.parquet"
                if file_path.exists():
                    artifacts.append(file_path)
                    self.manifest.add_artifact(
                        name=f"clean_data_{symbol}",
                        file_path=file_path,
                        stage="data_cleaning",
                        metadata={'symbol': symbol}
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

    def _run_labeling(self) -> StageResult:
        """Stage 4: Triple-Barrier Labeling."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 4: Triple-Barrier Labeling")
        self.logger.info("="*70)

        try:
            from labeling import main as apply_labels
            apply_labels()

            artifacts = []
            for symbol in self.config.symbols:
                file_path = self.config.final_data_dir / f"{symbol}_labeled.parquet"
                if file_path.exists():
                    artifacts.append(file_path)
                    self.manifest.add_artifact(
                        name=f"labeled_data_{symbol}",
                        file_path=file_path,
                        stage="labeling",
                        metadata={'symbol': symbol}
                    )

            end_time = datetime.now()
            return StageResult(
                stage_name="labeling",
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts
            )

        except Exception as e:
            self.logger.error(f"Labeling failed: {e}")
            self.logger.error(traceback.format_exc())
            return StageResult(
                stage_name="labeling",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )

    def _run_create_splits(self) -> StageResult:
        """Stage 5: Create Train/Val/Test Splits."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 5: Create Train/Val/Test Splits")
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

    def _run_generate_report(self) -> StageResult:
        """Stage 6: Generate Completion Report."""
        start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("STAGE 6: Generate Completion Report")
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
