"""
Tests for learning rate finder.

Tests the LRFinder class that implements the learning rate range test
(Smith 2017) to find optimal learning rates for neural networks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.neural.lr_finder import (
    LRFinder,
    LRFinderResult,
    find_lr_for_model,
)


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestLRFinder:
    """Tests for LRFinder class."""

    @pytest.fixture
    def sample_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample training data."""
        n_samples = 500
        n_features = 10
        n_classes = 3

        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, n_classes, (n_samples,))

        return X, y

    @pytest.fixture
    def train_loader(self, sample_data: tuple) -> DataLoader:
        """Create training DataLoader."""
        X, y = sample_data
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    @pytest.fixture
    def model(self) -> SimpleNet:
        """Create simple model."""
        return SimpleNet(input_size=10, hidden_size=32, output_size=3)

    def test_find_returns_result(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """find() should return LRFinderResult."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        result = finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=50,
        )

        assert isinstance(result, LRFinderResult)
        assert len(result.lrs) > 0
        assert len(result.losses) > 0
        assert len(result.lrs) == len(result.losses)
        assert result.suggested_lr > 0
        assert result.method in ("steepest", "minimum", "valley", "fallback")

    def test_suggested_lr_reasonable(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """Suggested LR should be within tested range."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        result = finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=50,
        )

        # Suggested LR should be reasonable (not at extremes unless that's optimal)
        assert result.suggested_lr >= result.min_lr_tested
        assert result.suggested_lr <= result.max_lr_tested * 10  # Allow some margin

    def test_state_restored_after_find(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """Model state should be restored after find()."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        # Get initial state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=50,
        )

        # Check state is restored
        final_state = model.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], final_state[key]), f"State mismatch for {key}"

    def test_early_stopping_on_divergence(
        self, model: SimpleNet, train_loader: DataLoader
    ) -> None:
        """Should stop early if loss diverges."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        result = finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=100.0,  # Very high end LR to trigger divergence
            num_iter=200,
            diverge_threshold=4.0,
        )

        # Should have stopped early due to divergence
        assert result.num_iterations < 200

    def test_result_to_dict(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """LRFinderResult should convert to dict."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        result = finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=50,
        )

        d = result.to_dict()

        assert "lrs" in d
        assert "losses" in d
        assert "suggested_lr" in d
        assert "method" in d
        assert "num_iterations" in d

    def test_plot_without_error(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """plot() should not raise errors."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=50,
        )

        # Should not raise (will save to file instead of showing)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lr_finder.png"
            try:
                finder.plot(save_path=str(save_path))
                # If matplotlib is available, file should exist
                # If not, this will just pass
            except ImportError:
                pass  # matplotlib not available

    def test_plot_requires_find_first(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """plot() should raise if find() not called."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        with pytest.raises(RuntimeError, match="Must run find"):
            finder.plot()

    def test_result_property(self, model: SimpleNet, train_loader: DataLoader) -> None:
        """result property should return latest result."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            use_amp=False,
        )

        # Before find()
        assert finder.result is None

        result = finder.find(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=50,
        )

        # After find()
        assert finder.result is result


class TestFindLRForModel:
    """Tests for find_lr_for_model convenience function."""

    @pytest.fixture
    def train_loader(self) -> DataLoader:
        """Create training DataLoader."""
        X = torch.randn(200, 10)
        y = torch.randint(0, 3, (200,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def test_returns_float(self, train_loader: DataLoader) -> None:
        """Should return a float learning rate."""
        model = SimpleNet(input_size=10, hidden_size=32, output_size=3)

        lr = find_lr_for_model(
            model=model,
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=30,
            device="cpu",
            use_amp=False,
        )

        assert isinstance(lr, float)
        assert lr > 0

    def test_custom_criterion(self, train_loader: DataLoader) -> None:
        """Should accept custom criterion."""
        model = SimpleNet(input_size=10, hidden_size=32, output_size=3)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        lr = find_lr_for_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=30,
            device="cpu",
            use_amp=False,
        )

        assert isinstance(lr, float)
        assert lr > 0


class TestLRFinderResult:
    """Tests for LRFinderResult dataclass."""

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        result = LRFinderResult(
            lrs=[1e-6, 1e-5, 1e-4],
            losses=[1.0, 0.9, 0.8],
            suggested_lr=1e-5,
            method="steepest",
            min_lr_tested=1e-6,
            max_lr_tested=1e-4,
            num_iterations=3,
            best_loss=0.8,
        )

        d = result.to_dict()

        assert d["lrs"] == [1e-6, 1e-5, 1e-4]
        assert d["losses"] == [1.0, 0.9, 0.8]
        assert d["suggested_lr"] == 1e-5
        assert d["method"] == "steepest"

    def test_metadata(self) -> None:
        """Should store extra metadata."""
        result = LRFinderResult(
            lrs=[1e-6],
            losses=[1.0],
            suggested_lr=1e-6,
            method="fallback",
            min_lr_tested=1e-6,
            max_lr_tested=1e-6,
            num_iterations=1,
            best_loss=1.0,
            metadata={"model": "lstm", "batch_size": 32},
        )

        assert result.metadata["model"] == "lstm"
        assert result.metadata["batch_size"] == 32
