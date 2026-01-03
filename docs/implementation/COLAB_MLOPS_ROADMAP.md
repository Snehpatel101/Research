# Google Colab MLOps Implementation Roadmap

## Overview

This roadmap outlines the implementation steps to enable Google Colab training for the ML factory. The goal is to add Colab support **without breaking existing CLI/local workflows**.

---

## Phase 1: Core Infrastructure (Priority 1)

### 1.1 Checkpoint Manager ✅

**Status:** Complete

**Files:**
- `colab_notebooks/utils/checkpoint_manager.py`

**Features:**
- Auto-save to Google Drive every 30 minutes
- W&B artifact upload
- Resume from latest checkpoint
- Cleanup old checkpoints (keep last 3)

---

### 1.2 Colab Setup Utilities ✅

**Status:** Complete

**Files:**
- `colab_notebooks/utils/colab_setup.py`

**Features:**
- Auto-mount Google Drive
- Clone/update repository
- Install dependencies
- GPU detection
- W&B authentication
- Environment summary

---

### 1.3 BaseModel State Management

**Status:** TODO

**Files to modify:**
- `src/models/base.py`

**Required changes:**
```python
class BaseModel(ABC):
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get model state for checkpointing."""
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state from checkpoint."""
        pass
```

**Implementation for each model:**
- XGBoost: `model.get_booster().save_raw()`
- LightGBM: `model.booster_.save_model()`
- CatBoost: `model.save_model()`
- LSTM/GRU/TCN: `model.state_dict()` (PyTorch)
- Transformers: `model.save_pretrained()`

**Estimate:** 2-3 hours

---

### 1.4 PipelineRunner Checkpoint Support

**Status:** TODO

**Files to modify:**
- `src/pipeline/runner.py`

**Required changes:**
```python
class PipelineRunner:
    def run(
        self,
        start_phase: int = 1,
        checkpoint_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Run pipeline with optional checkpointing.

        Args:
            start_phase: Phase to start from (for resume)
            checkpoint_callback: Called after each phase with (phase_num, state)
        """
        for phase_num in range(start_phase, 8):
            result = self._run_phase(phase_num)

            # Call checkpoint callback
            if checkpoint_callback:
                checkpoint_callback(phase_num, result.to_dict())
```

**Estimate:** 1-2 hours

---

## Phase 2: Notebook Templates (Priority 1)

### 2.1 Data Pipeline Notebook ✅

**Status:** Complete

**Files:**
- `colab_notebooks/01_data_pipeline.ipynb`

**Features:**
- Load data from Drive
- Resume from checkpoint
- Run phases 1-5 with auto-save
- Copy results to Drive

---

### 2.2 Tabular Training Notebook ✅

**Status:** Complete

**Files:**
- `colab_notebooks/02_train_tabular.ipynb`

**Features:**
- Train XGBoost/LightGBM/CatBoost
- Auto-checkpoint every 30 min
- W&B experiment tracking
- Save to Drive + W&B

---

### 2.3 Sequence Training Notebook

**Status:** TODO

**Files:**
- `colab_notebooks/03_train_sequence.ipynb`

**Features:**
- Train LSTM/GRU/TCN
- Epoch-level checkpointing
- GPU memory monitoring
- Early stopping with resume

**Estimate:** 1 hour (copy from 02_train_tabular.ipynb, adapt for sequence models)

---

### 2.4 Advanced Training Notebook

**Status:** TODO

**Files:**
- `colab_notebooks/04_train_advanced.ipynb`

**Features:**
- Train PatchTST/TFT/iTransformer
- Multi-stream data loading
- Mixed precision training
- Gradient checkpointing (for memory)

**Estimate:** 1-2 hours

---

### 2.5 Ensemble Training Notebook

**Status:** TODO

**Files:**
- `colab_notebooks/05_train_ensemble.ipynb`

**Features:**
- Load base models from W&B
- Generate OOF predictions
- Train stacking meta-learner
- Heterogeneous ensemble support

**Estimate:** 1-2 hours

---

## Phase 3: W&B Integration (Priority 1)

### 3.1 Add W&B to Trainer

**Status:** TODO

**Files to modify:**
- `src/models/trainer.py`

**Required changes:**
```python
class ModelTrainer:
    def __init__(
        self,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.wandb_project = wandb_project

        if self.wandb_project:
            import wandb
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                config=self.config,
            )

    def train(self, ...):
        # Log metrics to W&B
        if self.wandb_run:
            self.wandb_run.log({"train_loss": loss, "val_loss": val_loss})
```

**Estimate:** 2-3 hours

---

### 3.2 Add W&B Callbacks

**Status:** TODO

**Files to create:**
- `src/models/callbacks/wandb_callback.py`

**Features:**
- Log training metrics in real-time
- Upload model artifacts
- Log hyperparameters
- Log system metrics (GPU, CPU, RAM)

**Estimate:** 2-3 hours

---

## Phase 4: Data Management (Priority 2)

### 4.1 DVC Integration

**Status:** Partially complete (config files created)

**Files:**
- `.dvc/config` ✅
- `.dvc/.gitignore` ✅

**TODO:**
- Add DVC commands to notebooks
- Create DVC pipeline stages
- Document DVC workflow

**Estimate:** 2-3 hours

---

### 4.2 Drive Sync Utilities

**Status:** TODO

**Files to create:**
- `colab_notebooks/utils/drive_sync.py`

**Features:**
```python
def sync_to_drive(local_path: Path, drive_path: Path):
    """Copy local files to Google Drive with progress bar."""

def sync_from_drive(drive_path: Path, local_path: Path):
    """Copy Drive files to local with resume support."""
```

**Estimate:** 1-2 hours

---

## Phase 5: Monitoring & Alerts (Priority 2)

### 5.1 GPU Monitoring

**Status:** TODO

**Files to create:**
- `colab_notebooks/utils/gpu_monitor.py`

**Features:**
```python
class GPUMonitor:
    def __init__(self, log_interval: int = 60):
        """Monitor GPU utilization, VRAM, temperature."""

    def start(self):
        """Start background monitoring thread."""

    def get_stats(self) -> dict:
        """Get current GPU stats."""
```

**Estimate:** 1-2 hours

---

### 5.2 Session Time Manager

**Status:** TODO

**Files to create:**
- `colab_notebooks/utils/session_manager.py`

**Features:**
```python
class SessionManager:
    def __init__(self, session_limit_hours: float = 12.0):
        """Track session time and auto-save before timeout."""

    def time_remaining(self) -> float:
        """Hours remaining before timeout."""

    def register_checkpoint_callback(self, callback: Callable):
        """Auto-trigger checkpoint before timeout."""
```

**Estimate:** 1-2 hours

---

## Phase 6: Documentation (Priority 2)

### 6.1 Colab User Guide

**Status:** TODO

**Files to create:**
- `docs/guides/COLAB_TRAINING.md`

**Contents:**
- Setup instructions
- Notebook workflow
- Checkpoint/resume guide
- Troubleshooting common issues
- Best practices

**Estimate:** 2-3 hours

---

### 6.2 README Updates

**Status:** TODO

**Files to modify:**
- `README.md`
- `colab_notebooks/README.md`

**Contents:**
- Add Colab badge
- Quick start for Colab users
- Link to notebooks

**Estimate:** 30 min

---

## Phase 7: Testing & Validation (Priority 3)

### 7.1 Notebook Execution Tests

**Status:** TODO

**Files to create:**
- `tests/test_colab_notebooks.py`

**Features:**
- Use `papermill` to execute notebooks programmatically
- Verify outputs match expected results
- Test checkpoint/resume logic

**Estimate:** 2-3 hours

---

### 7.2 Integration Tests

**Status:** TODO

**Files to create:**
- `tests/integration/test_colab_pipeline.py`

**Features:**
- End-to-end pipeline test in Colab-like environment
- Mock Google Drive
- Test W&B integration

**Estimate:** 3-4 hours

---

## Implementation Timeline

| Phase | Priority | Estimated Time | Dependencies |
|-------|----------|----------------|--------------|
| 1.1-1.2 | P1 | ✅ Complete | None |
| 1.3 | P1 | 2-3 hours | None |
| 1.4 | P1 | 1-2 hours | 1.3 |
| 2.1-2.2 | P1 | ✅ Complete | 1.1-1.4 |
| 2.3-2.5 | P1 | 3-5 hours | 2.1-2.2 |
| 3.1-3.2 | P1 | 4-6 hours | None |
| 4.1-4.2 | P2 | 3-5 hours | None |
| 5.1-5.2 | P2 | 2-4 hours | None |
| 6.1-6.2 | P2 | 2-4 hours | All P1 |
| 7.1-7.2 | P3 | 5-7 hours | All P1, P2 |

**Total estimated time:** 22-36 hours (~3-5 days)

---

## Success Criteria

1. ✅ Auto-checkpoint to Drive every 30 min
2. ✅ W&B experiment tracking enabled
3. ✅ Resume training from checkpoint after disconnect
4. ✅ Zero data loss on disconnect (Drive + W&B backup)
5. Phase-based notebook workflow functional
6. Multi-session ensemble training works
7. Complete documentation for Colab users
8. All notebooks execute without errors

---

## Post-Implementation Optimization

### Optional Enhancements (Post-MVP)

1. **Prefect Cloud Integration** - Workflow orchestration
2. **Papermill Parameterization** - Programmatic notebook execution
3. **Great Expectations** - Data validation
4. **Cloud Storage (GCS)** - Alternative to Drive (faster, but costs money)
5. **Slack/Email Notifications** - Training completion alerts
6. **Model Registry UI** - Web interface for model browsing

---

## Notes

- **Backward compatibility:** All changes must be backward compatible with existing CLI workflows
- **No breaking changes:** Existing scripts must continue to work
- **Optional dependencies:** W&B, DVC should be optional (graceful degradation)
- **Testing:** All new code must have unit tests
