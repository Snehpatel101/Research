"""Tests for reproducibility controls."""

import random

import numpy as np
import pytest
import torch

from src.core.reproducibility import (
    ReproducibilityConfig,
    ReproducibilityInfo,
    ensure_reproducibility,
    get_reproducibility_info,
    get_worker_init_fn,
    set_all_seeds,
)


class TestReproducibilityConfig:
    """Tests for ReproducibilityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReproducibilityConfig()
        assert config.seed == 42
        assert config.deterministic is False
        assert config.benchmark is True
        assert config.warn_only is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReproducibilityConfig(seed=123, deterministic=True)
        assert config.seed == 123
        assert config.deterministic is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ReproducibilityConfig(seed=99)
        d = config.to_dict()
        assert d["seed"] == 99
        assert "deterministic" in d
        assert "benchmark" in d


class TestSetAllSeeds:
    """Tests for set_all_seeds function."""

    def test_python_random_seeded(self):
        """Test that Python random is seeded."""
        set_all_seeds(42)
        val1 = random.random()
        set_all_seeds(42)
        val2 = random.random()
        assert val1 == val2

    def test_numpy_random_seeded(self):
        """Test that NumPy random is seeded."""
        set_all_seeds(42)
        val1 = np.random.rand()
        set_all_seeds(42)
        val2 = np.random.rand()
        assert val1 == val2

    def test_torch_random_seeded(self):
        """Test that PyTorch random is seeded."""
        set_all_seeds(42)
        val1 = torch.rand(1).item()
        set_all_seeds(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_random_seeded(self):
        """Test that CUDA random is seeded."""
        set_all_seeds(42)
        val1 = torch.cuda.FloatTensor(1).uniform_().item()
        set_all_seeds(42)
        val2 = torch.cuda.FloatTensor(1).uniform_().item()
        assert val1 == val2

    def test_returns_reproducibility_info(self):
        """Test that function returns ReproducibilityInfo."""
        info = set_all_seeds(42)
        assert isinstance(info, ReproducibilityInfo)
        assert info.seed == 42
        assert info.python_random_seed_set is True
        assert info.numpy_seed_set is True
        assert info.torch_seed_set is True

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different values."""
        set_all_seeds(42)
        val1 = random.random()
        set_all_seeds(99)
        val2 = random.random()
        assert val1 != val2

    def test_negative_seed_raises(self):
        """Test that negative seed raises ValueError."""
        with pytest.raises(ValueError):
            set_all_seeds(-1)

    def test_deterministic_mode(self):
        """Test deterministic mode setting."""
        info = set_all_seeds(42, deterministic=True)
        assert info.cudnn_deterministic is True or not info.cuda_available
        # Reset to non-deterministic for other tests
        set_all_seeds(42, deterministic=False)


class TestGetReproducibilityInfo:
    """Tests for get_reproducibility_info function."""

    def test_returns_info_after_seeding(self):
        """Test info after seeds are set."""
        set_all_seeds(42)
        info = get_reproducibility_info()
        assert info.seed == 42
        assert info.torch_seed_set is True

    def test_contains_version_info(self):
        """Test that version info is included."""
        set_all_seeds(42)
        info = get_reproducibility_info()
        assert info.torch_version != "not installed"
        assert isinstance(info.cuda_available, bool)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        set_all_seeds(42)
        info = get_reproducibility_info()
        d = info.to_dict()
        assert "seed" in d
        assert "torch_version" in d
        assert "timestamp" in d


class TestEnsureReproducibility:
    """Tests for ensure_reproducibility function."""

    def test_sets_seed_if_provided(self):
        """Test that seed is set if provided."""
        info = ensure_reproducibility(seed=99)
        assert info.seed == 99

    def test_uses_existing_seed_if_none(self):
        """Test that existing seed is kept if none provided."""
        set_all_seeds(42)
        info = ensure_reproducibility(seed=None)
        # Should keep existing seed
        assert info.seed == 42

    def test_uses_default_if_no_seed_set(self):
        """Test default seed when no seed was previously set."""
        # Note: This test may be affected by other tests that set seeds
        # The function should return current seed if already set
        info = ensure_reproducibility()
        assert info.seed is not None


class TestGetWorkerInitFn:
    """Tests for get_worker_init_fn function."""

    def test_returns_callable(self):
        """Test that function returns a callable."""
        fn = get_worker_init_fn(42)
        assert callable(fn)

    def test_worker_init_sets_seeds(self):
        """Test that worker init function sets seeds."""
        fn = get_worker_init_fn(42)
        fn(0)  # Worker 0
        val1 = random.random()

        fn = get_worker_init_fn(42)
        fn(0)  # Worker 0 again
        val2 = random.random()

        assert val1 == val2

    def test_different_workers_different_seeds(self):
        """Test that different workers get different seeds."""
        fn = get_worker_init_fn(42)

        fn(0)
        val_worker_0 = random.random()

        fn(1)
        val_worker_1 = random.random()

        assert val_worker_0 != val_worker_1


class TestReproducibilityInfo:
    """Tests for ReproducibilityInfo dataclass."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        info = ReproducibilityInfo(
            seed=42,
            python_random_seed_set=True,
            numpy_seed_set=True,
            torch_seed_set=True,
            cuda_seed_set=False,
            cudnn_deterministic=False,
            cudnn_benchmark=True,
            torch_version="2.0.0",
            cuda_available=False,
            cuda_version=None,
            cudnn_version=None,
        )
        d = info.to_dict()
        assert d["seed"] == 42
        assert d["python_random_seed_set"] is True
        assert d["torch_version"] == "2.0.0"
        assert "timestamp" in d


class TestReproducibleTraining:
    """Integration tests for reproducible training."""

    def test_reproducible_model_initialization(self):
        """Test that model initialization is reproducible."""
        set_all_seeds(42)
        model1 = torch.nn.Linear(10, 3)
        weights1 = model1.weight.clone()

        set_all_seeds(42)
        model2 = torch.nn.Linear(10, 3)
        weights2 = model2.weight.clone()

        assert torch.allclose(weights1, weights2)

    def test_reproducible_training_step(self):
        """Test that a training step is reproducible."""
        set_all_seeds(42)

        model1 = torch.nn.Linear(10, 3)
        x = torch.randn(5, 10)
        y = torch.randint(0, 3, (5,))
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)

        output1 = model1(x)
        loss1 = torch.nn.CrossEntropyLoss()(output1, y)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        final_weights1 = model1.weight.clone()

        # Reset and repeat
        set_all_seeds(42)

        model2 = torch.nn.Linear(10, 3)
        x = torch.randn(5, 10)  # Same random data
        y = torch.randint(0, 3, (5,))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

        output2 = model2(x)
        loss2 = torch.nn.CrossEntropyLoss()(output2, y)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        final_weights2 = model2.weight.clone()

        assert torch.allclose(final_weights1, final_weights2)
