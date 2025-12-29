"""
Stage 9: Report Generation.

Pipeline wrapper for completion report generation.
"""
import json
import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.pipeline.utils import StageResult, create_failed_result, create_stage_result

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def generate_report_content(
    config: 'PipelineConfig',
    combined_df: pd.DataFrame,
    split_config: dict[str, Any],
    feature_cols: list[str],
    label_stats: dict[int, dict[str, int]],
    stage_results: dict[str, StageResult]
) -> str:
    """
    Generate the report markdown content.

    Args:
        config: Pipeline configuration
        combined_df: Combined labeled DataFrame
        split_config: Split configuration dictionary
        feature_cols: List of feature column names
        label_stats: Label distribution statistics
        stage_results: Results from all stages

    Returns:
        Markdown formatted report string
    """
    from src.pipeline.utils import StageStatus

    report = f'''# Phase 1 Completion Report
## Ensemble Price Prediction System

**Run ID:** {config.run_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

Phase 1 successfully processed raw OHLCV data through the complete pipeline:
- Data Cleaning (1-min to 5-min resampling)
- Feature Engineering ({len(feature_cols)} technical features)
- Triple-Barrier Labeling ({len(config.label_horizons)} horizons: {', '.join(map(str, config.label_horizons))} bars)
- Train/Val/Test Splits (with purging & embargo)

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | {len(combined_df):,} |
| **Symbols** | {', '.join(config.symbols)} |
| **Date Range** | {combined_df['datetime'].min()} to {combined_df['datetime'].max()} |
| **Resolution** | {config.bar_resolution} |
| **Features** | {len(feature_cols)} |

---

## Label Distribution

'''
    for horizon in config.label_horizons:
        stats = label_stats.get(horizon, {})
        total = sum(stats.values()) if stats else len(combined_df)
        if total == 0:
            total = 1
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
    for stage_name, result in stage_results.items():
        status_icon = "PASS" if result.status == StageStatus.COMPLETED else "FAIL"
        report += f'''### [{status_icon}] {stage_name.replace('_', ' ').title()}
- **Status:** {result.status.value}
- **Duration:** {result.duration_seconds:.2f} seconds
- **Artifacts:** {len(result.artifacts)}

'''

    sample_features = feature_cols[:5] if len(feature_cols) >= 5 else feature_cols

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
feature_cols = {sample_features}  # Example features
X_train = train_df[feature_cols].values
y_train = train_df['label_h5'].values  # For 5-bar horizon
sample_weights = train_df['sample_weight_h5'].values
```

---

*Phase 1 Complete - Ready for Phase 2*
'''
    return report


def run_generate_report(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest',
    stage_results: dict[str, StageResult]
) -> StageResult:
    """
    Stage 9: Generate Completion Report.

    Creates a comprehensive markdown report summarizing:
    - Dataset overview
    - Label distributions
    - Split statistics
    - Pipeline execution summary
    - Phase 2 preparation guidance

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs
        stage_results: Results from all previous stages

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 9: Generate Completion Report")
    logger.info("=" * 70)

    try:
        combined_path = config.final_data_dir / "combined_final_labeled.parquet"
        combined_df = pd.read_parquet(combined_path)

        with open(config.splits_dir / "split_config.json") as f:
            split_config = json.load(f)

        excluded_prefixes = (
            'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
            'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
            'fwd_return_log_', 'time_to_hit_'
        )
        feature_cols = [
            c for c in combined_df.columns
            if c not in [
                'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
            ]
            and not any(c.startswith(prefix) for prefix in excluded_prefixes)
        ]

        label_stats = {}
        for horizon in config.label_horizons:
            col = f'label_h{horizon}'
            if col in combined_df.columns:
                counts = combined_df[col].value_counts().sort_index()
                label_stats[horizon] = {
                    'short': int(counts.get(-1, 0)),
                    'neutral': int(counts.get(0, 0)),
                    'long': int(counts.get(1, 0))
                }

        report = generate_report_content(
            config=config,
            combined_df=combined_df,
            split_config=split_config,
            feature_cols=feature_cols,
            label_stats=label_stats,
            stage_results=stage_results
        )

        # Run-scoped output for reproducibility
        report_path = (
            config.run_artifacts_dir / f"PHASE1_COMPLETION_REPORT_{config.run_id}.md"
        )
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to: {report_path}")

        manifest.add_artifact(
            name="completion_report",
            file_path=report_path,
            stage="generate_report",
            metadata={
                'total_samples': len(combined_df),
                'num_features': len(feature_cols)
            }
        )

        return create_stage_result(
            stage_name="generate_report",
            start_time=start_time,
            artifacts=[report_path],
            metadata={
                'total_samples': len(combined_df),
                'num_features': len(feature_cols)
            }
        )

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="generate_report",
            start_time=start_time,
            error=str(e)
        )
