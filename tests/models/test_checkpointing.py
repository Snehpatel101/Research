"""Tests for training checkpoint management."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.models.neural.checkpointing import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointMetadata,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CheckpointConfig()
        assert config.interval_epochs == 10
        assert config.keep_n_best == 3
        assert config.save_optimizer is True
        assert config.metric_mode == "min"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CheckpointConfig(
            checkpoint_dir=Path("/tmp/test"),
            interval_epochs=5,
            keep_n_best=5,
        )
        assert config.interval_epochs == 5
        assert config.keep_n_best == 5

    def test_invalid_interval_raises(self):
        """Test that invalid interval raises error."""
        with pytest.raises(ValueError):
            CheckpointConfig(interval_epochs=0)

    def test_invalid_keep_n_raises(self):
        """Test that invalid keep_n raises error."""
        with pytest.raises(ValueError):
            CheckpointConfig(keep_n_best=0)

    def test_invalid_metric_mode_raises(self):
        """Test that invalid metric_mode raises error."""
        with pytest.raises(ValueError):
            CheckpointConfig(metric_mode="invalid")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CheckpointConfig(interval_epochs=5)
        d = config.to_dict()
        assert d["interval_epochs"] == 5
        assert "checkpoint_dir" in d
        assert "metric_mode" in d


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = CheckpointMetadata(
            epoch=5,
            metrics={"val_loss": 0.5},
            timestamp="2024-01-01_12:00:00",
            config={"lr": 0.001},
            checkpoint_path="/tmp/ckpt.pt",
        )
        d = metadata.to_dict()
        assert d["epoch"] == 5
        assert d["metrics"]["val_loss"] == 0.5

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "epoch": 10,
            "metrics": {"val_loss": 0.3},
            "timestamp": "2024-01-01_12:00:00",
            "config": {},
            "checkpoint_path": "/tmp/ckpt.pt",
            "is_best": True,
        }
        metadata = CheckpointMetadata.from_dict(d)
        assert metadata.epoch == 10
        assert metadata.is_best is True


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def tmp_ckpt_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        return ckpt_dir

    @pytest.fixture
    def manager(self, tmp_ckpt_dir):
        """Create checkpoint manager."""
        config = CheckpointConfig(
            checkpoint_dir=tmp_ckpt_dir,
            interval_epochs=2,
            keep_n_best=3,
        )
        return CheckpointManager(config)

    @pytest.fixture
    def model_and_optimizer(self):
        """Create model and optimizer for testing."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return model, optimizer, scheduler

    def test_save_checkpoint(self, manager, model_and_optimizer, tmp_ckpt_dir):
        """Test saving a checkpoint."""
        model, optimizer, scheduler = model_and_optimizer
        metrics = {"val_loss": 0.5, "train_loss": 0.6}

        path = manager.save_checkpoint(
            model, optimizer, scheduler, epoch=5, metrics=metrics
        )

        assert path.exists()
        assert path.suffix == ".pt"
        assert manager.num_checkpoints == 1

    def test_maybe_save_checkpoint_at_interval(self, manager, model_and_optimizer):
        """Test maybe_save_checkpoint at interval."""
        model, optimizer, scheduler = model_and_optimizer
        metrics = {"val_loss": 0.5}

        # Epoch 0 (1st epoch) - not at interval
        path = manager.maybe_save_checkpoint(
            model, optimizer, scheduler, epoch=0, metrics=metrics
        )
        assert path is None

        # Epoch 1 (2nd epoch) - at interval (interval=2)
        path = manager.maybe_save_checkpoint(
            model, optimizer, scheduler, epoch=1, metrics=metrics
        )
        assert path is not None

    def test_load_checkpoint(self, manager, model_and_optimizer, tmp_ckpt_dir):
        """Test loading a checkpoint."""
        model, optimizer, scheduler = model_and_optimizer
        original_weights = model.fc.weight.clone()

        # Save checkpoint
        metrics = {"val_loss": 0.5}
        path = manager.save_checkpoint(
            model, optimizer, scheduler, epoch=5, metrics=metrics
        )

        # Modify model weights
        with torch.no_grad():
            model.fc.weight.fill_(0.0)

        # Create new model and load
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        result = manager.load_checkpoint(
            path, new_model, new_optimizer
        )

        assert result["epoch"] == 5
        assert result["metrics"]["val_loss"] == 0.5
        assert torch.allclose(new_model.fc.weight, original_weights)

    def test_best_checkpoint_tracking(self, manager, model_and_optimizer):
        """Test best checkpoint is tracked correctly."""
        model, optimizer, scheduler = model_and_optimizer

        # Save with val_loss=0.5
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=1,
            metrics={"val_loss": 0.5}
        )
        assert manager.best_metric == 0.5

        # Save with better val_loss=0.3
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=2,
            metrics={"val_loss": 0.3}
        )
        assert manager.best_metric == 0.3

        # Save with worse val_loss=0.7
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=3,
            metrics={"val_loss": 0.7}
        )
        assert manager.best_metric == 0.3  # Still 0.3

    def test_get_best_checkpoint(self, manager, model_and_optimizer, tmp_ckpt_dir):
        """Test getting best checkpoint path."""
        model, optimizer, scheduler = model_and_optimizer

        # Save checkpoints with different losses
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=1,
            metrics={"val_loss": 0.5}
        )
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=2,
            metrics={"val_loss": 0.3}
        )

        best_path = manager.get_best_checkpoint()
        assert best_path is not None
        assert best_path.name == "best.pt"
        assert best_path.exists()

    def test_get_latest_checkpoint(self, manager, model_and_optimizer):
        """Test getting latest checkpoint path."""
        model, optimizer, scheduler = model_and_optimizer

        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=1,
            metrics={"val_loss": 0.5}
        )
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch=10,
            metrics={"val_loss": 0.6}
        )

        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "epoch0010" in str(latest)

    def test_checkpoint_pruning(self, tmp_path):
        """Test that old checkpoints are pruned."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()

        config = CheckpointConfig(
            checkpoint_dir=ckpt_dir,
            interval_epochs=1,
            keep_n_best=2,  # Only keep 2 best
        )
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save 5 checkpoints with different losses
        for i, loss in enumerate([0.5, 0.3, 0.7, 0.2, 0.4]):
            manager.save_checkpoint(
                model, optimizer, None, epoch=i,
                metrics={"val_loss": loss}
            )

        # Should only have 2 checkpoints (best losses: 0.2, 0.3)
        assert manager.num_checkpoints == 2

        # Check the right ones were kept
        checkpoints = manager.list_checkpoints()
        losses = [c.metrics["val_loss"] for c in checkpoints]
        assert 0.2 in losses
        assert 0.3 in losses

    def test_checkpoint_index_persistence(self, tmp_path):
        """Test checkpoint index is saved and loaded."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()

        config = CheckpointConfig(checkpoint_dir=ckpt_dir)

        # Create manager and save checkpoint
        manager1 = CheckpointManager(config)
        model = SimpleModel()
        manager1.save_checkpoint(
            model, None, None, epoch=5,
            metrics={"val_loss": 0.5}
        )

        # Create new manager - should load existing index
        manager2 = CheckpointManager(config)
        assert manager2.num_checkpoints == 1
        assert manager2.best_metric == 0.5

    def test_checkpoint_contains_optimizer_state(self, manager, model_and_optimizer, tmp_ckpt_dir):
        """Test that optimizer state is saved."""
        model, optimizer, scheduler = model_and_optimizer

        # Do some optimization steps
        x = torch.randn(5, 10)
        y = torch.randint(0, 3, (5,))
        for _ in range(3):
            loss = nn.CrossEntropyLoss()(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        path = manager.save_checkpoint(
            model, optimizer, scheduler, epoch=3,
            metrics={"val_loss": 0.5}
        )

        # Load checkpoint and verify optimizer state
        checkpoint = torch.load(path, weights_only=False)
        assert "optimizer_state_dict" in checkpoint
        assert "step" in checkpoint["optimizer_state_dict"]["state"][0]

    def test_extra_state_saved(self, manager, model_and_optimizer):
        """Test that extra state is saved and loaded."""
        model, optimizer, scheduler = model_and_optimizer
        extra = {"custom_key": "custom_value", "epoch_losses": [0.5, 0.4, 0.3]}

        path = manager.save_checkpoint(
            model, optimizer, scheduler, epoch=5,
            metrics={"val_loss": 0.3},
            extra_state=extra,
        )

        result = manager.load_checkpoint(path, model, optimizer)
        assert result["extra_state"]["custom_key"] == "custom_value"
        assert result["extra_state"]["epoch_losses"] == [0.5, 0.4, 0.3]
