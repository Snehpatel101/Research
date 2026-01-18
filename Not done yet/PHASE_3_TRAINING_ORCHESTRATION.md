# PHASE 3: TRAINING ORCHESTRATION - Unified Training Pipeline

**Status:** PLANNING
**Created:** 2026-01-17
**Purpose:** Single entry point for all training modes, CV methods, and OOF generation

---

## Overview

Phase 3 creates a unified training orchestration system where:
1. `TrainingOrchestrator` is the single entry point for all training
2. Four training modes are supported: standard, walk-forward, regime-aware, meta-labeling
3. Four CV methods are integrated: PurgedKFold, CPCV, PBO, WalkForward
4. OOF generation handles both tabular (2D) and sequence (3D) models

**Design Principles:**
1. **Single Entry Point** - All training flows through `TrainingOrchestrator`
2. **Mode Agnostic** - Same interface regardless of training mode
3. **CV Integration** - Proper purging/embargo across all modes
4. **OOF Alignment** - Heterogeneous models produce aligned predictions

---

## Data Flow Diagram

```
                          TrainingOrchestrator.train()
                                    |
                                    v
                    +-------------------------------+
                    |     Select Training Mode      |
                    +-------------------------------+
                    |                               |
          +---------+---------+---------+---------+
          |         |         |         |
          v         v         v         v
      Standard  Walk-Fwd  Regime-   Meta-
      Training  Training  Aware     Labeling
          |         |         |         |
          v         v         v         v
    +-------------------------------------------+
    |         Select CV Method                   |
    +-------------------------------------------+
    | PurgedKFold | CPCV | PBO | WalkForwardCV  |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |         OOF Generation                     |
    +-------------------------------------------+
    | Tabular (2D)    |    Sequence (3D)        |
    | CoreOOFGenerator | SequenceOOFGenerator   |
    +-------------------------------------------+
                          |
                          v
    +-------------------------------------------+
    |         Alignment & Stacking               |
    | OOFPrediction -> StackingDataset          |
    +-------------------------------------------+
                          |
                          v
                   Phase 4: Meta-Learners
```

---

## Task 3.1: TrainingOrchestrator - Single Entry Point

### File: `src/training/orchestrator.py`

```python
"""
TrainingOrchestrator - Master controller for unified ML training system.

This is the SINGLE ENTRY POINT for all training in the ML Factory.
All training modes, CV methods, and model types flow through here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

from src.core.types import TrainingMode, CVMethod
from src.models import Trainer, TrainerConfig
from src.models.registry import ModelRegistry
from src.cross_validation import PurgedKFold, PurgedKFoldConfig
from src.cross_validation.oof_generator import OOFGenerator
from src.training.modes.walk_forward import WalkForwardTrainer, WalkForwardTrainerConfig
from src.training.modes.regime_aware import RegimeAwareTrainer, RegimeAwareConfig
from src.training.modes.meta_labeling import MetaLabelingTrainer, MetaLabelingConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """
    Configuration for training orchestration.

    Attributes:
        mode: Training mode (standard, walk_forward, regime_aware, meta_labeling)
        cv_method: Cross-validation method (purged_kfold, cpcv, pbo, walk_forward)
        models: List of model names to train
        horizons: List of label horizons
        n_splits: Number of CV folds (default 5 for boosting, 3 for neural)
        purge_bars: Bars to purge before test set (default 60)
        embargo_bars: Bars to embargo after test set (default 1440)
        generate_oof: Whether to generate OOF predictions
        build_ensemble: Whether to build ensemble from trained models
        meta_learner: Meta-learner for ensemble (ridge_meta, mlp_meta, etc.)
        output_dir: Output directory for artifacts
    """
    mode: TrainingMode = TrainingMode.STANDARD
    cv_method: CVMethod = CVMethod.PURGED_KFOLD
    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    horizons: list[int] = field(default_factory=lambda: [20])
    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440
    generate_oof: bool = True
    build_ensemble: bool = False
    meta_learner: str = "ridge_meta"
    output_dir: Path = field(default_factory=lambda: Path("outputs/training"))

    # Mode-specific configs
    walk_forward_config: WalkForwardTrainerConfig | None = None
    regime_config: RegimeAwareConfig | None = None
    meta_labeling_config: MetaLabelingConfig | None = None


class TrainingOrchestrator:
    """
    Master controller for unified ML training system.

    SINGLE ENTRY POINT for all training:
    - Standard training with PurgedKFold
    - Walk-forward validation
    - Regime-aware training
    - Meta-labeling (Lopez de Prado)

    Usage:
        >>> config = OrchestrationConfig(
        ...     mode=TrainingMode.WALK_FORWARD,
        ...     models=["xgboost", "lightgbm", "lstm"],
        ...     horizons=[20],
        ... )
        >>> orchestrator = TrainingOrchestrator(config)
        >>> results = orchestrator.train(container)

    Unified Interface:
        >>> orchestrator.train(models=["xgboost"], mode="walk_forward")
    """

    def __init__(self, config: OrchestrationConfig) -> None:
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = config.output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CV
        self._cv = self._create_cv()

        # Initialize OOF generator
        self._oof_generator = OOFGenerator(self._cv)

        # Results storage
        self.results: dict[str, Any] = {}
        self.trained_models: dict[str, Trainer] = {}
        self.oof_predictions: dict[str, Any] = {}

        logger.info(f"Initialized TrainingOrchestrator")
        logger.info(f"  Mode: {config.mode}")
        logger.info(f"  CV: {config.cv_method}")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  Output: {self.output_dir}")

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{self.config.mode.value}_{timestamp}"

    def _create_cv(self) -> PurgedKFold:
        """Create cross-validator based on config."""
        cv_config = PurgedKFoldConfig(
            n_splits=self.config.n_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )
        return PurgedKFold(cv_config)

    def train(
        self,
        container: TimeSeriesDataContainer,
        models: list[str] | None = None,
        mode: str | TrainingMode | None = None,
    ) -> dict[str, Any]:
        """
        Execute training pipeline.

        This is the UNIFIED INTERFACE for all training.

        Args:
            container: TimeSeriesDataContainer with train/val/test data
            models: Override models from config (optional)
            mode: Override mode from config (optional)

        Returns:
            Dict with training results
        """
        # Override config if provided
        models = models or self.config.models
        if mode is not None:
            mode = TrainingMode(mode) if isinstance(mode, str) else mode
        else:
            mode = self.config.mode

        logger.info("=" * 60)
        logger.info(f"TRAINING ORCHESTRATION: {mode.value.upper()}")
        logger.info("=" * 60)

        # Route to appropriate training mode
        if mode == TrainingMode.STANDARD:
            results = self._train_standard(container, models)
        elif mode == TrainingMode.WALK_FORWARD:
            results = self._train_walk_forward(container, models)
        elif mode == TrainingMode.REGIME_AWARE:
            results = self._train_regime_aware(container, models)
        elif mode == TrainingMode.META_LABELING:
            results = self._train_meta_labeling(container, models)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        # Generate OOF predictions if requested
        if self.config.generate_oof and mode == TrainingMode.STANDARD:
            self._generate_oof_predictions(container, models)

        # Build ensemble if requested
        if self.config.build_ensemble and len(models) > 1:
            self._build_ensemble(container)

        self._save_results()

        return {
            "run_id": self.run_id,
            "mode": mode.value,
            "model_results": self.results,
            "oof_predictions": self.oof_predictions,
            "output_dir": str(self.output_dir),
        }

    def _train_standard(self, container, models: list[str]) -> dict[str, Any]:
        """Standard training with PurgedKFold CV."""
        results = {}
        for horizon in self.config.horizons:
            horizon_results = {}
            for model_name in models:
                trainer_config = TrainerConfig(
                    model_name=model_name,
                    horizon=horizon,
                    output_dir=self.output_dir / f"h{horizon}",
                )
                trainer = Trainer(trainer_config)
                training_results = trainer.run(container)
                horizon_results[model_name] = training_results
                self.trained_models[f"{model_name}_h{horizon}"] = trainer
            results[f"horizon_{horizon}"] = horizon_results
        self.results = results
        return results

    def _train_walk_forward(self, container, models: list[str]) -> dict[str, Any]:
        """Walk-forward validation training."""
        wf_config = self.config.walk_forward_config or WalkForwardTrainerConfig()
        # Implementation delegates to WalkForwardTrainer
        return {}

    def _train_regime_aware(self, container, models: list[str]) -> dict[str, Any]:
        """Regime-aware training."""
        regime_config = self.config.regime_config or RegimeAwareConfig()
        # Implementation delegates to RegimeAwareTrainer
        return {}

    def _train_meta_labeling(self, container, models: list[str]) -> dict[str, Any]:
        """Meta-labeling training (Lopez de Prado)."""
        meta_config = self.config.meta_labeling_config or MetaLabelingConfig(
            primary_model=models[0] if models else "xgboost",
            meta_model="logistic",
        )
        # Implementation delegates to MetaLabelingTrainer
        return {}

    def _generate_oof_predictions(self, container, models: list[str]) -> None:
        """Generate OOF predictions for all models."""
        logger.info("Generating OOF predictions...")
        # Implementation uses OOFGenerator
        pass

    def _build_ensemble(self, container) -> None:
        """Build ensemble from OOF predictions."""
        logger.info(f"Building ensemble with meta-learner: {self.config.meta_learner}")
        # Implementation builds stacking dataset
        pass

    def _save_results(self) -> None:
        """Save results to disk."""
        import json
        results_path = self.output_dir / "results.json"
        serializable = {
            "run_id": self.run_id,
            "mode": self.config.mode.value,
            "models": self.config.models,
            "horizons": self.config.horizons,
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)


__all__ = ["TrainingOrchestrator", "OrchestrationConfig"]
```

---

## Task 3.2: Training Modes Integration

### Mode Summary Table

| Mode | Use Case | Key Feature |
|------|----------|-------------|
| `standard` | Default training | PurgedKFold CV with OOF generation |
| `walk_forward` | Production simulation | Expanding/rolling windows, time-aware |
| `regime_aware` | Adaptive strategies | Separate models per volatility/trend regime |
| `meta_labeling` | Bet sizing | Primary + meta-model for position sizing |

---

## Task 3.3: Cross-Validation Methods

### CV Method Selection Guide

```python
def get_cv_for_use_case(use_case: str, base_config: PurgedKFoldConfig) -> Any:
    """Select appropriate CV method based on use case."""
    if use_case == "training":
        return PurgedKFold(base_config)
    elif use_case == "validation":
        return CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2))
    elif use_case == "overfitting_check":
        return CombinatorialPurgedCV(CPCVConfig(n_groups=10, n_test_groups=2))
    elif use_case == "production":
        return WalkForwardEvaluator(WalkForwardConfig(n_windows=5))
    else:
        raise ValueError(f"Unknown use case: {use_case}")
```

### CV Fold Structure Diagram

```
PURGED K-FOLD (5 folds):
|====Train====|PURGE|==Test==|EMBARGO|====Train====|
|    Fold 1   |  60 | Fold 2 | 1440  |   Fold 3-5  |

WALK-FORWARD (expanding):
Window 1: |----Train----|--Test--|
Window 2: |------Train------|--Test--|
Window 3: |--------Train--------|--Test--|

CPCV (6 groups, 2 test):
Path 1:  |Train|Train|TEST|TEST|Train|Train|
Path 2:  |Train|TEST|Train|TEST|Train|Train|
...etc (15 combinations)
```

---

## Task 3.4: OOF Generation System

### OOFPrediction Data Structure

```python
@dataclass
class OOFPrediction:
    """Out-of-fold predictions for a single model."""
    model_name: str
    predictions: pd.DataFrame
    fold_info: list[dict]
    coverage: float = 1.0
    original_indices: np.ndarray | None = None
    sequence_length: int | None = None
    n_total_samples: int | None = None

    @property
    def alignment_offset(self) -> int:
        """Offset from start to first valid prediction."""
        if self.sequence_length:
            return self.sequence_length - 1
        return 0
```

---

## Task 3.5: Heterogeneous OOF Alignment

### File: `src/cross_validation/oof_alignment.py`

```python
"""OOF alignment for heterogeneous stacking."""

class OOFAlignmentValidator:
    """Validate and align OOF predictions from heterogeneous models."""

    def compute_alignment(self, oof_predictions: dict[str, OOFPrediction]) -> dict:
        """Compute alignment parameters."""
        max_offset = 0
        for model_name, oof_pred in oof_predictions.items():
            max_offset = max(max_offset, oof_pred.alignment_offset)
        return {"max_offset": max_offset}

    def align_predictions(self, oof_predictions: dict, alignment_info: dict) -> dict:
        """Align all OOF predictions to common valid range."""
        max_offset = alignment_info["max_offset"]
        aligned = {}
        for model_name, oof_pred in oof_predictions.items():
            probs = oof_pred.get_probabilities()
            model_offset = oof_pred.alignment_offset
            local_start = max_offset - model_offset
            aligned[model_name] = probs[local_start:]
        return aligned
```

---

## Task 3.6: Unified Interface Examples

```python
# Example 1: Standard Training
config = OrchestrationConfig(
    mode=TrainingMode.STANDARD,
    models=["xgboost", "lightgbm", "random_forest"],
    horizons=[20],
    generate_oof=True,
    build_ensemble=True,
)
orchestrator = TrainingOrchestrator(config)
results = orchestrator.train(container)

# Example 2: Walk-Forward Validation
config = OrchestrationConfig(
    mode=TrainingMode.WALK_FORWARD,
    models=["xgboost", "lstm"],
    horizons=[20],
)
orchestrator = TrainingOrchestrator(config)
results = orchestrator.train(container)

# Example 3: Simplified Unified Interface
orchestrator = TrainingOrchestrator(OrchestrationConfig())
results = orchestrator.train(
    container,
    models=["xgboost", "lstm", "transformer"],
    mode="walk_forward",
)
```

---

## Implementation Checklist

### Task 3.1: TrainingOrchestrator
- [ ] Create `src/training/orchestrator.py`
- [ ] `OrchestrationConfig` dataclass
- [ ] `TrainingOrchestrator` class
- [ ] `train()` unified entry point
- [ ] Route to appropriate training mode

### Task 3.2: Training Modes
- [ ] WalkForwardTrainer integration
- [ ] RegimeAwareTrainer integration
- [ ] MetaLabelingTrainer integration

### Task 3.3: CV Methods
- [ ] PurgedKFold (existing)
- [ ] CombinatorialPurgedCV (existing)
- [ ] PBO computation (existing)
- [ ] WalkForwardEvaluator (existing)

### Task 3.4: OOF Generation
- [ ] OOFGenerator unified interface
- [ ] CoreOOFGenerator for tabular models
- [ ] SequenceOOFGenerator for sequence models

### Task 3.5: OOF Alignment
- [ ] OOFAlignmentValidator
- [ ] `compute_alignment()` method
- [ ] `align_predictions()` method

### Task 3.6: Integration Testing
- [ ] Standard training end-to-end
- [ ] Walk-forward training end-to-end
- [ ] Heterogeneous OOF alignment

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-17 |
| Purpose | Unified training orchestration for all modes |
| Related Docs | PHASE_1B_LABELING_OPTIMIZATION.md, PHASE_4_META_LEARNERS.md |
| Depends On | PHASE_0_FOUNDATION.md, PHASE_2_ADAPTER_INTEGRATION.md |
