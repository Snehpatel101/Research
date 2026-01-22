# Codebase Recommendations: Ensemble Price Prediction
**Date**: January 22, 2026  
**Based On**: Architectural Flow Analysis (CODEBASE_REVIEW.md)  
**Focus**: Actionable improvements for architecture, flow, and cohesion

---

## Executive Summary

This document provides **prioritized, actionable recommendations** to improve the ensemble-price-prediction codebase from its current **B- grade (83/100)** to **A- grade (90/100)**.

**Current State**: Strong financial ML foundation with architectural cohesion issues  
**Target State**: Production-ready system with clean architecture and scalable flow  
**Estimated Effort**: 2-3 days for critical fixes, 1-2 weeks for full improvements

---

## Priority Matrix

| Priority | Issue | Impact | Effort | Quick Win? |
|----------|-------|--------|--------|------------|
| 🔴 **CRITICAL** | Dual orchestration paths | High | 4 hours | ⚠️ Medium |
| 🔴 **CRITICAL** | Duplicated triple-barrier | High | 2 hours | ✅ Yes |
| 🔴 **CRITICAL** | God orchestrator refactor | High | 2 days | ❌ No |
| 🟡 **HIGH** | Implicit data contracts | Medium | 1 day | ⚠️ Medium |
| 🟡 **HIGH** | Module consolidation | Medium | 1 day | ❌ No |
| 🟢 **MEDIUM** | Add diversity analysis | Low | 4 hours | ✅ Yes |
| 🟢 **MEDIUM** | Parallel training | Medium | 1 day | ⚠️ Medium |
| 🟢 **LOW** | Enhanced bet sizing | Low | 4 hours | ✅ Yes |

---

## 🔴 CRITICAL PRIORITY (Do First)

### 1. Consolidate Orchestration Entry Point

**Problem**: Two orchestration paths create confusion
```
User sees:    MLPipeline (src/orchestrator.py)
Reality:      UnifiedTrainingOrchestrator (src/training/unified_orchestrator.py)
```

**Impact**: 
- New developers don't know which to use
- Documentation inconsistency
- Phase artifacts may be bypassed

**Recommendation**: Make the relationship explicit

#### Option A: Document the Hierarchy (RECOMMENDED - 1 hour)

```python
# src/orchestrator.py - Add clear docstring

class MLPipeline:
    """
    THE primary entry point for the complete ML pipeline.
    
    This is a high-level orchestrator that runs all 9 phases:
    1-4: Data preparation (features, labels, splits)
    5-6: Training and ensemble building (delegates to UnifiedTrainingOrchestrator)
    7-9: Evaluation, backtest, bundling
    
    For TRAINING-ONLY workflows, you can use UnifiedTrainingOrchestrator directly.
    
    Architecture:
        MLPipeline
          ├─> Data Phases (1-4): In-house pipeline stages
          ├─> Training Phase (5-6): UnifiedTrainingOrchestrator
          └─> Post-Training (7-9): Evaluation, backtest, bundling
    
    Example (Full Pipeline):
        >>> config = PipelineConfig(symbol="MES", models=["xgboost", "lstm"])
        >>> result = MLPipeline(config).run()  # Runs all 9 phases
    
    Example (Training Only):
        >>> from src.training import UnifiedTrainingOrchestrator
        >>> orchestrator = UnifiedTrainingOrchestrator(config)
        >>> result = orchestrator.train(df)  # Skips data prep phases
    """
```

#### Option B: Rename for Clarity (2 hours)

```python
# Rename to clarify hierarchy
MLPipeline → FullPipeline  (includes data prep)
UnifiedTrainingOrchestrator → TrainingPipeline (training only)

# Update all imports
from src.orchestrator import FullPipeline
from src.training import TrainingPipeline
```

**Acceptance Criteria**:
- [ ] Docstrings clearly explain when to use each
- [ ] README updated with both usage patterns
- [ ] Examples in docs show both paths

---

### 2. Unify Triple-Barrier Implementations

**Problem**: Two implementations risk label divergence

```
Implementation 1: src/labeling/triple_barrier.py
Implementation 2: src/pipeline/stages/labeling/triple_barrier.py
```

**Impact**: 
- Updates to one may not propagate to other
- Training/inference label mismatch
- Maintenance burden (2× the code)

**Recommendation**: Single source of truth

#### Step 1: Choose Authoritative Implementation (30 min)

```bash
# Compare the two implementations
diff src/labeling/triple_barrier.py \
     src/pipeline/stages/labeling/triple_barrier.py

# Decision: Keep src/labeling/triple_barrier.py
# Reason: More complete, has Numba optimization, better tested
```

#### Step 2: Make Pipeline Stage a Thin Wrapper (1 hour)

```python
# src/pipeline/stages/labeling/triple_barrier.py
"""
Triple-Barrier Labeling Pipeline Stage.

IMPORTANT: This is a THIN WRAPPER around src.labeling.triple_barrier.
Do NOT duplicate implementation logic here.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.labeling.triple_barrier import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    LabelingResult,
)
from src.pipeline.utils import StageResult, StageStatus


@dataclass
class TripleBarrierStageConfig:
    """Configuration for triple-barrier pipeline stage."""
    horizons: list[int] = None
    upper_mult: float = 2.0
    lower_mult: float = 2.0
    atr_period: int = 14
    apply_transaction_costs: bool = True
    symbol: str = "MES"
    
    def to_labeling_config(self, horizon: int) -> TripleBarrierConfig:
        """Convert to TripleBarrierConfig."""
        return TripleBarrierConfig(
            upper_mult=self.upper_mult,
            lower_mult=self.lower_mult,
            horizon=horizon,
            atr_period=self.atr_period,
            apply_transaction_costs=self.apply_transaction_costs,
            symbol=self.symbol,
        )


class TripleBarrierLabelingStage:
    """
    Pipeline stage for triple-barrier labeling.
    
    This stage delegates to src.labeling.triple_barrier.TripleBarrierLabeler
    for the actual labeling logic. It handles:
    - Loading feature-engineered data
    - Applying labeling for multiple horizons
    - Saving labeled datasets
    """
    
    def __init__(self, config: TripleBarrierStageConfig):
        self.config = config
        self.horizons = config.horizons or [10, 15, 20]
    
    def run(self, df: pd.DataFrame, output_dir: Path) -> StageResult:
        """
        Run triple-barrier labeling for all horizons.
        
        Args:
            df: Feature-engineered OHLCV DataFrame
            output_dir: Directory to save labeled datasets
        
        Returns:
            StageResult with success status and artifacts
        """
        try:
            artifacts = {}
            
            for horizon in self.horizons:
                # Create labeler with horizon-specific config
                labeling_config = self.config.to_labeling_config(horizon)
                labeler = TripleBarrierLabeler(labeling_config)
                
                # Generate labels (delegates to authoritative implementation)
                result: LabelingResult = labeler.create_labels(df)
                
                # Add labels to DataFrame
                label_col = f"label_h{horizon}"
                df[label_col] = result.labels
                
                # Save labeled dataset
                output_path = output_dir / f"labeled_h{horizon}.parquet"
                df.to_parquet(output_path)
                artifacts[f"labeled_h{horizon}"] = output_path
            
            return StageResult(
                status=StageStatus.SUCCESS,
                artifacts=artifacts,
                message=f"Triple-barrier labeling complete for {len(self.horizons)} horizons"
            )
            
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Triple-barrier labeling failed: {e}"
            )
```

#### Step 3: Add Integration Test (30 min)

```python
# tests/integration/test_triple_barrier_consistency.py
"""
Ensure both triple-barrier entry points produce identical labels.
"""
import pytest
import pandas as pd
import numpy as np

from src.labeling.triple_barrier import TripleBarrierLabeler, TripleBarrierConfig
from src.pipeline.stages.labeling.triple_barrier import (
    TripleBarrierLabelingStage,
    TripleBarrierStageConfig
)


def test_triple_barrier_implementations_match():
    """Both implementations should produce identical labels."""
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="5min")
    df = pd.DataFrame({
        "open": np.random.randn(1000).cumsum() + 100,
        "high": np.random.randn(1000).cumsum() + 102,
        "low": np.random.randn(1000).cumsum() + 98,
        "close": np.random.randn(1000).cumsum() + 100,
        "volume": np.random.randint(100, 1000, 1000),
        "atr_14": np.random.uniform(0.5, 2.0, 1000),
    }, index=dates)
    
    horizon = 20
    
    # Method 1: Direct labeler
    config1 = TripleBarrierConfig(
        upper_mult=2.0,
        lower_mult=2.0,
        horizon=horizon,
        atr_period=14,
    )
    labeler = TripleBarrierLabeler(config1)
    result1 = labeler.create_labels(df)
    
    # Method 2: Pipeline stage
    config2 = TripleBarrierStageConfig(
        horizons=[horizon],
        upper_mult=2.0,
        lower_mult=2.0,
        atr_period=14,
    )
    stage = TripleBarrierLabelingStage(config2)
    result2 = stage.run(df.copy(), output_dir=Path("/tmp"))
    
    df_labeled = pd.read_parquet(result2.artifacts[f"labeled_h{horizon}"])
    labels2 = df_labeled[f"label_h{horizon}"].values
    
    # Assert identical labels
    assert np.array_equal(result1.labels, labels2), \
        "Triple-barrier implementations produce different labels!"
```

**Acceptance Criteria**:
- [ ] Only ONE implementation of core algorithm exists
- [ ] Pipeline stage is <50 lines (thin wrapper)
- [ ] Integration test passes
- [ ] Both entry points documented

---

### 3. Refactor God Orchestrator

**Problem**: `UnifiedTrainingOrchestrator` has 8 responsibilities (1,600+ LOC)

**Current Responsibilities**:
1. Route to training mode ✅ (should keep)
2. Manage data preparation ⚠️ (delegate to adapter layer)
3. Train individual models ⚠️ (extract to service)
4. Generate OOF predictions ⚠️ (extract to service)
5. Align OOF predictions ⚠️ (move to ensemble module)
6. Build ensembles ⚠️ (move to ensemble module)
7. Save artifacts ⚠️ (extract to persistence layer)
8. Optimize hyperparameters ⚠️ (extract to tuning service)

**Impact**:
- Hard to test individual components
- Changes ripple across unrelated functionality
- Difficult to extend without touching core class

**Recommendation**: Extract services (2 days effort)

#### New Architecture

```python
# src/training/services/__init__.py
"""Training services for modular orchestration."""

from .model_training import ModelTrainingService
from .oof_generation import OOFGenerationService
from .hyperparameter_tuning import HyperparameterTuningService
from .artifact_persistence import ArtifactManager

__all__ = [
    "ModelTrainingService",
    "OOFGenerationService",
    "HyperparameterTuningService",
    "ArtifactManager",
]
```

#### Service 1: Model Training Service

```python
# src/training/services/model_training.py
"""Service for training individual models."""

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
import pandas as pd

from src.adapters import PreparedData
from src.models.base import ModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingRequest:
    """Request to train a single model."""
    model_name: str
    horizon: int
    prepared_data: PreparedData
    optimize_hyperparams: bool = False


@dataclass
class ModelTrainingResult:
    """Result from training a single model."""
    model_name: str
    horizon: int
    trainer: Any
    metrics: dict[str, float]
    training_time_seconds: float
    n_features: int
    data_rank: int


class ModelTrainingService:
    """
    Service for training individual models.
    
    Responsibilities:
    - Create model trainer
    - Execute training
    - Return structured result
    
    Does NOT:
    - Prepare data (handled by adapters)
    - Generate OOF (handled by OOFGenerationService)
    - Save artifacts (handled by ArtifactManager)
    """
    
    def train_model(self, request: ModelTrainingRequest) -> ModelTrainingResult:
        """
        Train a single model.
        
        Args:
            request: ModelTrainingRequest with model config and data
        
        Returns:
            ModelTrainingResult with trained model and metrics
        """
        logger.info(f"Training {request.model_name} (horizon={request.horizon})")
        
        import time
        start = time.time()
        
        # Create trainer
        trainer = self._create_trainer(request)
        
        # Train
        prepared = request.prepared_data
        training_results = trainer.train(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
        )
        
        training_time = time.time() - start
        
        return ModelTrainingResult(
            model_name=request.model_name,
            horizon=request.horizon,
            trainer=trainer,
            metrics=training_results.get("metrics", {}),
            training_time_seconds=training_time,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
        )
    
    def _create_trainer(self, request: ModelTrainingRequest) -> ModelTrainer:
        """Create appropriate trainer for model."""
        # Import and instantiate trainer
        # (Existing logic from UnifiedTrainingOrchestrator)
        ...
```

#### Service 2: OOF Generation Service

```python
# src/training/services/oof_generation.py
"""Service for generating out-of-fold predictions."""

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
import pandas as pd

from src.cross_validation import OOFGenerator, OOFPrediction, PurgedKFold
from src.adapters import PreparedData

logger = logging.getLogger(__name__)


@dataclass
class OOFRequest:
    """Request to generate OOF predictions."""
    model_name: str
    horizon: int
    prepared_data: PreparedData
    cv_folds: int = 5


class OOFGenerationService:
    """
    Service for generating out-of-fold predictions.
    
    Responsibilities:
    - Create CV strategy
    - Generate OOF predictions via cross-validation
    - Return OOFPrediction object
    
    Does NOT:
    - Train the final model (handled by ModelTrainingService)
    - Align OOF predictions (handled by EnsembleOrchestrator)
    """
    
    def generate_oof(self, request: OOFRequest) -> OOFPrediction:
        """
        Generate out-of-fold predictions for a model.
        
        Args:
            request: OOFRequest with model config and data
        
        Returns:
            OOFPrediction with probabilities and metadata
        """
        logger.info(f"Generating OOF for {request.model_name}")
        
        # Create CV strategy
        cv = PurgedKFold(n_splits=request.cv_folds)
        
        # Flatten to 2D if needed (for OOF generation)
        X_train = self._flatten_if_needed(request.prepared_data.X_train)
        y_train = request.prepared_data.y_train
        
        # Convert to DataFrame for OOFGenerator
        X_train_df = pd.DataFrame(
            X_train,
            columns=request.prepared_data.feature_names
        )
        
        # Generate OOF predictions
        oof_generator = OOFGenerator(cv=cv)
        oof_predictions = oof_generator.generate(
            model_name=request.model_name,
            X=X_train_df,
            y=y_train,
        )
        
        return oof_predictions
    
    def _flatten_if_needed(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D/4D arrays to 2D for OOF generation."""
        if X.ndim == 2:
            return X
        elif X.ndim == 3:  # (samples, seq_len, features)
            return X.reshape(X.shape[0], -1)
        elif X.ndim == 4:  # (samples, timeframes, seq_len, features)
            return X.reshape(X.shape[0], -1)
        else:
            raise ValueError(f"Unexpected array rank: {X.ndim}")
```

#### Refactored Orchestrator (Thin)

```python
# src/training/unified_orchestrator.py (AFTER refactor)
"""
UnifiedTrainingOrchestrator - Thin coordinator for training workflows.

NOW focuses ONLY on:
- Routing to training modes
- Coordinating services
- Managing workflow state

Delegates to:
- ModelTrainingService: Individual model training
- OOFGenerationService: Out-of-fold prediction generation
- EnsembleOrchestrator: Ensemble building
- ArtifactManager: Saving results
"""

from src.training.services import (
    ModelTrainingService,
    OOFGenerationService,
    ArtifactManager,
)
from src.models.ensemble import EnsembleOrchestrator


class UnifiedTrainingOrchestrator:
    """
    Thin orchestrator that coordinates training workflows.
    
    Reduced from 1,600 lines to ~400 lines by extracting services.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = config.output_dir / self.run_id
        
        # Initialize services
        self.model_service = ModelTrainingService()
        self.oof_service = OOFGenerationService()
        self.ensemble_orchestrator = EnsembleOrchestrator(config)
        self.artifact_manager = ArtifactManager(self.output_dir)
        
        # State tracking
        self._model_results: dict[str, ModelTrainingResult] = {}
        self._oof_predictions: dict[str, OOFPrediction] = {}
    
    def train(self, df: pd.DataFrame) -> TrainingRunResult:
        """
        Train models based on configured mode.
        
        Now delegates to services instead of implementing everything.
        """
        mode = self.config.training_mode
        
        if mode == TrainingMode.STANDARD:
            return self._train_standard(df)
        elif mode == TrainingMode.WALK_FORWARD:
            return self._train_walk_forward(df)
        elif mode == TrainingMode.REGIME_AWARE:
            return self._train_regime_aware(df)
        elif mode == TrainingMode.META_LABELING:
            return self._train_meta_labeling(df)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
    
    def _train_standard(self, df: pd.DataFrame) -> TrainingRunResult:
        """Standard training mode (delegates to services)."""
        # For each model/horizon
        for model_name in self.config.models:
            for horizon in self.config.horizons:
                # 1. Prepare data (existing adapter logic)
                prepared = self._prepare_data(df, model_name, horizon)
                
                # 2. Train model (DELEGATE)
                request = ModelTrainingRequest(
                    model_name=model_name,
                    horizon=horizon,
                    prepared_data=prepared,
                    optimize_hyperparams=self.config.optimize_hyperparams,
                )
                result = self.model_service.train_model(request)
                
                # 3. Generate OOF (DELEGATE)
                if self.config.build_ensemble:
                    oof_request = OOFRequest(
                        model_name=model_name,
                        horizon=horizon,
                        prepared_data=prepared,
                        cv_folds=self.config.cv_folds,
                    )
                    oof = self.oof_service.generate_oof(oof_request)
                    self._oof_predictions[f"{model_name}_h{horizon}"] = oof
                
                # Store result
                self._model_results[f"{model_name}_h{horizon}"] = result
        
        # 4. Build ensemble (DELEGATE)
        ensemble_result = None
        if self.config.build_ensemble:
            ensemble_result = self.ensemble_orchestrator.train_from_oof(
                self._oof_predictions
            )
        
        # 5. Save artifacts (DELEGATE)
        self.artifact_manager.save_results(
            model_results=self._model_results,
            ensemble_result=ensemble_result,
        )
        
        return TrainingRunResult(...)
```

**Benefits After Refactor**:
- ✅ Each service <300 lines (Single Responsibility)
- ✅ Easy to test in isolation
- ✅ Clear separation of concerns
- ✅ Orchestrator reduced to ~400 lines (coordination only)
- ✅ Services are reusable across different workflows

**Acceptance Criteria**:
- [ ] Services extracted to `src/training/services/`
- [ ] Each service has single responsibility
- [ ] Unit tests for each service
- [ ] Orchestrator reduced to <500 lines
- [ ] All existing functionality preserved

---

## 🟡 HIGH PRIORITY (Do Second)

### 4. Introduce Explicit Dataset Contract

**Problem**: Implicit conventions for data passing between stages

**Current Issues**:
- Column names assumed (`label`, `y_true`, etc.)
- Index assumptions not documented
- OOF alignment relies on implicit index overlap

**Recommendation**: Define explicit data contract

```python
# src/core/data_contract.py
"""
Explicit data contracts for all pipeline stages.

This eliminates implicit assumptions about column names, indices, and metadata.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class DatasetContract:
    """
    Explicit contract for datasets passed between pipeline stages.
    
    This is THE standard format for all data transformations.
    Every stage should accept and return DatasetContract objects.
    
    Attributes:
        features: Feature DataFrame (columns = feature names)
        labels: Label Series (index must match features)
        indices: Explicit index (for OOF alignment)
        label_end_times: End time of label horizon (for purge calculation)
        split: Which split this belongs to ("train", "val", "test")
        metadata: Arbitrary metadata (symbol, horizon, etc.)
    """
    # Core data
    features: pd.DataFrame
    labels: pd.Series
    
    # Index tracking (EXPLICIT)
    indices: pd.Index
    label_end_times: Optional[pd.Series] = None
    
    # Metadata
    split: str = "train"  # "train", "val", "test"
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate contract consistency."""
        # Check feature/label length match
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Feature/label length mismatch: "
                f"{len(self.features)} != {len(self.labels)}"
            )
        
        # Check index match
        if not self.features.index.equals(self.labels.index):
            raise ValueError("Feature and label indices must match")
        
        # Set indices if not provided
        if self.indices is None:
            self.indices = self.features.index
    
    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return len(self.features)
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.features.columns)
    
    @property
    def feature_names(self) -> list[str]:
        """List of feature column names."""
        return list(self.features.columns)
    
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays (for model training)."""
        return self.features.values, self.labels.values
    
    def filter_by_indices(self, indices: pd.Index) -> "DatasetContract":
        """
        Filter dataset to specific indices.
        
        Useful for OOF alignment where we need common samples.
        """
        mask = self.indices.isin(indices)
        
        return DatasetContract(
            features=self.features.iloc[mask],
            labels=self.labels.iloc[mask],
            indices=self.indices[mask],
            label_end_times=self.label_end_times.iloc[mask] if self.label_end_times is not None else None,
            split=self.split,
            metadata=self.metadata.copy(),
        )


# Usage Example:

# Before (implicit):
def train_model(X_train, y_train):
    # What are the indices? What's the split? Unknown!
    pass

# After (explicit):
def train_model(dataset: DatasetContract):
    assert dataset.split == "train"
    X, y = dataset.to_numpy()
    indices = dataset.indices  # Explicit!
    # ...
```

**Acceptance Criteria**:
- [ ] `DatasetContract` class created
- [ ] All pipeline stages updated to use contract
- [ ] OOF alignment uses `filter_by_indices()`
- [ ] No more implicit column name assumptions

---

### 5. Consolidate Top-Level Modules

**Problem**: 20+ top-level modules create unclear boundaries

**Current Structure**:
```
src/
├── adapters/          # Data format
├── backtesting/       # Backtesting
├── cli/               # CLI
├── common/            # Utilities
├── config/            # Configuration
├── contracts/         # Interfaces
├── coordination/      # Pipeline coordination
├── core/              # Core types
├── cross_validation/  # CV strategies
├── evaluation/        # Evaluation
├── feature_selection/ # Feature selection
├── feature_store/     # Feature storage
├── features/          # Feature engineering
├── inference/         # Inference
├── labeling/          # Labeling
├── ml_pipeline/       # (duplicate?)
├── models/            # Models
├── monitoring/        # Monitoring
├── optimization/      # Optimization
├── pipeline/          # Data pipeline
├── training/          # Training
├── utils/             # More utilities
└── validation/        # Validation
```

**Recommendation**: Consolidate into 7 core domains

```
src/
├── core/               # Core types, contracts, configuration
│   ├── config.py
│   ├── contracts.py
│   └── constants.py
│
├── data/               # ALL data-related logic
│   ├── pipeline/       # (from old pipeline/)
│   ├── features/       # (from old features/)
│   ├── labeling/       # (from old labeling/)
│   └── adapters/       # (from old adapters/)
│
├── models/             # ALL model-related logic
│   ├── registry/       # Model plugin registry
│   ├── trainers/       # Training implementations
│   ├── ensemble/       # Ensemble methods
│   └── calibration/    # (from old models/calibration/)
│
├── validation/         # ALL validation logic
│   ├── cross_validation/  # (from old cross_validation/)
│   ├── evaluation/     # (from old evaluation/)
│   └── monitoring/     # (from old monitoring/)
│
├── optimization/       # Hyperparameter tuning
│   └── ...            # (keep as-is)
│
├── inference/          # Serving pipeline
│   └── ...            # (keep as-is)
│
└── cli/                # Command-line interface
    └── ...            # (keep as-is)
```

**Migration Plan** (1 day):

```bash
# 1. Create new structure
mkdir -p src/data/{pipeline,features,labeling,adapters}
mkdir -p src/models/{registry,trainers,ensemble,calibration}
mkdir -p src/validation/{cross_validation,evaluation,monitoring}

# 2. Move modules
mv src/pipeline/* src/data/pipeline/
mv src/features/* src/data/features/
mv src/labeling/* src/data/labeling/
mv src/adapters/* src/data/adapters/

# 3. Update imports (use sed or IDE refactor)
find src -name "*.py" -exec sed -i 's/from src.pipeline/from src.data.pipeline/g' {} \;
find src -name "*.py" -exec sed -i 's/from src.features/from src.data.features/g' {} \;
# ... etc for all moves

# 4. Update __init__.py files to re-export
# src/data/__init__.py
from .pipeline import *
from .features import *
from .labeling import *
from .adapters import *
```

**Acceptance Criteria**:
- [ ] 20+ modules reduced to 7 core domains
- [ ] All imports updated
- [ ] Tests still pass
- [ ] Documentation updated

---

## 🟢 MEDIUM PRIORITY (Do Third)

### 6. Add Ensemble Diversity Analysis

**Problem**: All models automatically included in ensemble

**Issue**: Highly correlated models add little value

**Recommendation**: Filter redundant models before ensembling

```python
# src/models/ensemble/diversity.py
"""
Diversity analysis for ensemble model selection.

Ensures we only ensemble models that provide unique perspectives.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


def compute_diversity_matrix(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise diversity (disagreement) between models.
    
    Uses Matthews Correlation Coefficient (MCC) between predictions.
    
    Args:
        predictions: List of prediction arrays (n_models, n_samples)
    
    Returns:
        Diversity matrix (n_models, n_models)
        Higher values = more diverse (less correlated)
    """
    n_models = len(predictions)
    diversity_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # MCC ranges from -1 to 1
            # Convert to diversity: diversity = 1 - abs(mcc)
            mcc = matthews_corrcoef(predictions[i], predictions[j])
            diversity = 1.0 - abs(mcc)
            
            diversity_matrix[i, j] = diversity
            diversity_matrix[j, i] = diversity
    
    # Diagonal is 0 (perfect correlation with self)
    np.fill_diagonal(diversity_matrix, 0.0)
    
    return diversity_matrix


def select_diverse_models(
    oof_predictions: dict[str, OOFPrediction],
    min_diversity: float = 0.3,
    max_models: int = 10,
) -> List[str]:
    """
    Select diverse subset of models for ensembling.
    
    Strategy:
    1. Start with best-performing model
    2. Iteratively add models that maximize diversity
    3. Stop when diversity threshold not met or max_models reached
    
    Args:
        oof_predictions: Dict of model_name -> OOFPrediction
        min_diversity: Minimum required diversity (0.0 to 1.0)
        max_models: Maximum models to select
    
    Returns:
        List of selected model names
    """
    if len(oof_predictions) <= max_models:
        return list(oof_predictions.keys())
    
    # Get predictions as arrays
    model_names = list(oof_predictions.keys())
    predictions = [oof.predictions for oof in oof_predictions.values()]
    
    # Compute diversity matrix
    diversity_matrix = compute_diversity_matrix(predictions)
    
    # Get accuracies for each model
    accuracies = np.array([
        oof.metrics.get("accuracy", 0.5)
        for oof in oof_predictions.values()
    ])
    
    # Start with best model
    selected_indices = [np.argmax(accuracies)]
    
    # Greedily add diverse models
    while len(selected_indices) < max_models:
        # Compute average diversity of each candidate to selected models
        avg_diversity = np.mean(
            diversity_matrix[selected_indices, :], axis=0
        )
        
        # Mask already selected
        avg_diversity[selected_indices] = -1.0
        
        # Find most diverse candidate
        next_idx = np.argmax(avg_diversity)
        
        # Check if meets diversity threshold
        if avg_diversity[next_idx] < min_diversity:
            break
        
        selected_indices.append(next_idx)
    
    selected_names = [model_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_names)}/{len(model_names)} diverse models:")
    for name in selected_names:
        print(f"  - {name}")
    
    return selected_names


# Integration with EnsembleOrchestrator:

class EnsembleOrchestrator:
    def train_from_oof(
        self,
        oof_predictions: dict[str, OOFPrediction],
        filter_diverse: bool = True,
        min_diversity: float = 0.3,
    ):
        """Train ensemble with optional diversity filtering."""
        
        if filter_diverse:
            # Select diverse subset
            selected_models = select_diverse_models(
                oof_predictions,
                min_diversity=min_diversity,
            )
            
            # Filter to selected models
            oof_predictions = {
                k: v for k, v in oof_predictions.items()
                if k in selected_models
            }
        
        # Continue with existing ensemble training logic
        ...
```

**Acceptance Criteria**:
- [ ] Diversity analysis implemented
- [ ] Integrated into `EnsembleOrchestrator`
- [ ] Configurable via `PipelineConfig`
- [ ] Logged which models were filtered out

---

### 7. Enable Parallel Model Training

**Problem**: Models trained sequentially (slow for 23 models)

**Current**:
```python
for model_name in models:
    result = train_model(model_name)  # Sequential
```

**Recommendation**: Parallel training with joblib

```python
# src/training/services/parallel_training.py
"""Parallel model training for faster experimentation."""

from joblib import Parallel, delayed
from typing import List
import logging

logger = logging.getLogger(__name__)


class ParallelTrainingService:
    """
    Service for training multiple models in parallel.
    
    Uses joblib for multiprocessing.
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        Args:
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.n_jobs = n_jobs
    
    def train_models_parallel(
        self,
        training_requests: List[ModelTrainingRequest],
        verbose: int = 10,
    ) -> List[ModelTrainingResult]:
        """
        Train multiple models in parallel.
        
        Args:
            training_requests: List of ModelTrainingRequest objects
            verbose: Verbosity level for joblib
        
        Returns:
            List of ModelTrainingResult objects
        """
        logger.info(
            f"Training {len(training_requests)} models in parallel "
            f"(n_jobs={self.n_jobs})"
        )
        
        # Train in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
            delayed(self._train_single)(request)
            for request in training_requests
        )
        
        return results
    
    def _train_single(self, request: ModelTrainingRequest) -> ModelTrainingResult:
        """Train a single model (called by joblib)."""
        from src.training.services import ModelTrainingService
        
        service = ModelTrainingService()
        return service.train_model(request)


# Integration in UnifiedTrainingOrchestrator:

def _train_standard(self, df: pd.DataFrame) -> TrainingRunResult:
    """Standard training with optional parallelization."""
    
    # Create all training requests
    requests = []
    for model_name in self.config.models:
        for horizon in self.config.horizons:
            prepared = self._prepare_data(df, model_name, horizon)
            request = ModelTrainingRequest(
                model_name=model_name,
                horizon=horizon,
                prepared_data=prepared,
            )
            requests.append(request)
    
    # Train in parallel if enabled
    if self.config.parallel_training:
        parallel_service = ParallelTrainingService(
            n_jobs=self.config.n_jobs
        )
        results = parallel_service.train_models_parallel(requests)
    else:
        # Sequential (existing behavior)
        results = [
            self.model_service.train_model(req)
            for req in requests
        ]
    
    # Store results
    for result in results:
        key = f"{result.model_name}_h{result.horizon}"
        self._model_results[key] = result
    
    ...
```

**Configuration**:
```python
# src/core/config.py
@dataclass
class PipelineConfig:
    # ... existing fields ...
    
    # Parallel training
    parallel_training: bool = True  # Enable by default
    n_jobs: int = -1  # Use all CPUs
```

**Acceptance Criteria**:
- [ ] Parallel training implemented
- [ ] Configurable via `PipelineConfig`
- [ ] Backwards compatible (can disable)
- [ ] Logs show parallel progress

---

## 🟢 LOW PRIORITY (Nice to Have)

### 8. Enhanced Bet Sizing Beyond Binary

**Problem**: Meta-labeling only does binary (trade/no-trade)

**Recommendation**: Variable position sizing

```python
# src/training/meta_labeling/bet_sizing.py
"""Enhanced bet sizing strategies."""

from enum import Enum
import numpy as np


class BetSizingStrategy(Enum):
    """Bet sizing strategies for meta-labeling."""
    BINARY = "binary"          # Current: trade or no-trade
    PROPORTIONAL = "proportional"  # Size ∝ probability
    KELLY = "kelly"            # Kelly Criterion
    RISK_PARITY = "risk_parity"  # Risk-adjusted sizing


def compute_bet_sizes(
    probabilities: np.ndarray,
    strategy: BetSizingStrategy = BetSizingStrategy.BINARY,
    threshold: float = 0.5,
    max_size: float = 1.0,
) -> np.ndarray:
    """
    Compute position sizes based on meta-model probabilities.
    
    Args:
        probabilities: P(correct) from meta-model [0, 1]
        strategy: Bet sizing strategy
        threshold: Minimum probability to trade
        max_size: Maximum position size (as fraction of capital)
    
    Returns:
        Position sizes [0, max_size]
    """
    if strategy == BetSizingStrategy.BINARY:
        # Current approach: 1.0 if prob > threshold, else 0.0
        return np.where(probabilities > threshold, max_size, 0.0)
    
    elif strategy == BetSizingStrategy.PROPORTIONAL:
        # Size proportional to confidence
        # Map [threshold, 1.0] → [0, max_size]
        sizes = np.maximum(0, probabilities - threshold) / (1.0 - threshold)
        return sizes * max_size
    
    elif strategy == BetSizingStrategy.KELLY:
        # Simplified Kelly Criterion
        # f* = (p * (b + 1) - 1) / b
        # where p = probability, b = odds
        # Assume b = 1 (even odds)
        kelly_fractions = 2 * probabilities - 1  # Ranges [-1, 1]
        kelly_fractions = np.maximum(0, kelly_fractions)  # Only positive
        return kelly_fractions * max_size
    
    elif strategy == BetSizingStrategy.RISK_PARITY:
        # Size inversely proportional to volatility
        # (Requires volatility as input - placeholder)
        raise NotImplementedError("Risk parity requires volatility input")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Usage in meta-labeling:

def predict_with_sizing(
    primary_predictions: np.ndarray,
    meta_probabilities: np.ndarray,
    strategy: BetSizingStrategy = BetSizingStrategy.KELLY,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict with variable bet sizing.
    
    Returns:
        directions: -1, 0, +1 (from primary model)
        sizes: [0, 1] (from meta-model + sizing strategy)
    """
    # Compute bet sizes
    sizes = compute_bet_sizes(meta_probabilities, strategy=strategy)
    
    # Apply sizes to directions
    # If size = 0, direction becomes 0 (no trade)
    final_directions = np.where(sizes > 0, primary_predictions, 0)
    
    return final_directions, sizes
```

**Acceptance Criteria**:
- [ ] Multiple sizing strategies implemented
- [ ] Configurable via `PipelineConfig`
- [ ] Backtesting uses variable position sizes
- [ ] Documented strategy differences

---

## Implementation Roadmap

### Week 1: Critical Fixes (🔴)

**Day 1-2**:
- [ ] Document orchestration hierarchy (1 hour)
- [ ] Unify triple-barrier implementations (2 hours)
- [ ] Add integration test for triple-barrier (30 min)

**Day 3-5**:
- [ ] Extract ModelTrainingService (1 day)
- [ ] Extract OOFGenerationService (1 day)
- [ ] Refactor orchestrator to use services (1 day)

### Week 2: High Priority (🟡)

**Day 6-7**:
- [ ] Create DatasetContract class (4 hours)
- [ ] Update pipeline stages to use contract (1 day)

**Day 8-10**:
- [ ] Consolidate module structure (1 day)
- [ ] Update all imports (1 day)
- [ ] Update documentation (4 hours)

### Week 3: Medium/Low Priority (🟢)

**Day 11**:
- [ ] Add diversity analysis (4 hours)
- [ ] Integrate with ensemble orchestrator (2 hours)

**Day 12**:
- [ ] Implement parallel training (4 hours)
- [ ] Test parallelization (2 hours)

**Day 13-14**:
- [ ] Enhanced bet sizing strategies (4 hours)
- [ ] Documentation and examples (1 day)

---

## Success Metrics

### Before Improvements
- **Architecture Grade**: B- (83/100)
- **Flow Clarity**: 6/10
- **Cohesion**: 6.5/10
- **Maintainability**: Medium

### After Improvements
- **Architecture Grade**: A- (90/100)
- **Flow Clarity**: 8/10 (clear orchestration, explicit contracts)
- **Cohesion**: 8/10 (services extracted, single responsibilities)
- **Maintainability**: High (testable, extensible)

---

## Effort Summary

| Priority | Total Effort | Quick Wins | High Impact |
|----------|--------------|------------|-------------|
| 🔴 Critical | 3 days | 1 (triple-barrier) | All 3 |
| 🟡 High | 3 days | 0 | 2/2 |
| 🟢 Medium/Low | 2 days | 2 (diversity, sizing) | 1/3 |
| **TOTAL** | **8 days** | **3/8 tasks** | **6/8 tasks** |

**Minimum Viable Improvement**: 
- Fix critical issues only (3 days)
- Achieves grade improvement to **B+ (87/100)**

**Recommended Path**:
- Complete critical + high priority (6 days)
- Achieves target grade of **A- (90/100)**

---

## Maintenance Guidelines

After implementing these improvements:

### 1. Enforce Single Responsibility
- No service should exceed 300 lines
- Each class should have one clear purpose

### 2. Prevent God Classes
- If a class reaches 500+ lines → extract services
- Maximum 5 public methods per orchestrator

### 3. Explicit Over Implicit
- Use `DatasetContract` for all data passing
- Document all assumptions in docstrings
- No "magic" column names

### 4. Test New Services
- Every new service needs unit tests
- Integration tests for service combinations
- Property-based tests for data transformations

### 5. Document Architectural Decisions
- Create ADR (Architecture Decision Record) for major changes
- Update flow diagrams when changing orchestration
- Keep this recommendations doc updated

---

**End of Recommendations**

**For Questions or Clarifications**: Refer to CODEBASE_REVIEW.md for detailed analysis of current architecture.
