"""Tests for OOM recovery manager."""

import pytest
import torch

from src.models.neural.oom_recovery import (
    OOMConfig,
    OOMEvent,
    OOMRecoveryManager,
    create_oom_manager,
)


class TestOOMConfig:
    """Tests for OOMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OOMConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.batch_reduction_factor == 0.5
        assert config.min_batch_size == 8

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OOMConfig(
            enabled=False,
            max_retries=5,
            batch_reduction_factor=0.7,
            min_batch_size=16,
        )
        assert config.enabled is False
        assert config.max_retries == 5
        assert config.batch_reduction_factor == 0.7
        assert config.min_batch_size == 16


class TestOOMRecoveryManager:
    """Tests for OOMRecoveryManager."""

    @pytest.fixture
    def manager(self):
        """Create OOM recovery manager."""
        return OOMRecoveryManager()

    @pytest.fixture
    def disabled_manager(self):
        """Create disabled OOM recovery manager."""
        config = OOMConfig(enabled=False)
        return OOMRecoveryManager(config)

    def test_is_oom_error_cuda(self, manager):
        """Test OOM error detection for CUDA errors."""
        oom_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert manager.is_oom_error(oom_error)

        oom_error2 = RuntimeError("out of memory")
        assert manager.is_oom_error(oom_error2)

    def test_is_oom_error_non_oom(self, manager):
        """Test that non-OOM errors are not detected."""
        other_error = RuntimeError("Some other error")
        assert not manager.is_oom_error(other_error)

        value_error = ValueError("Not a runtime error")
        assert not manager.is_oom_error(value_error)

    def test_handle_oom_reduces_batch_size(self, manager):
        """Test that handle_oom reduces batch size."""
        new_size = manager.handle_oom(256)
        assert new_size == 128  # 256 * 0.5

        new_size = manager.handle_oom(128)
        assert new_size == 64

    def test_handle_oom_respects_min_batch_size(self, manager):
        """Test that handle_oom respects minimum batch size."""
        # With min_batch_size=8, reducing 16 by 0.5 gives 8 (ok)
        new_size = manager.handle_oom(16)
        assert new_size == 8

        # But reducing 8 by 0.5 gives 4, which is below min
        manager.reset()
        new_size = manager.handle_oom(8)
        assert new_size is None

    def test_handle_oom_respects_max_retries(self, manager):
        """Test that handle_oom respects max retries."""
        # Default max_retries is 3
        assert manager.handle_oom(256) == 128  # Retry 1
        assert manager.handle_oom(128) == 64   # Retry 2
        assert manager.handle_oom(64) == 32    # Retry 3
        assert manager.handle_oom(32) is None  # Exceeds max retries

    def test_handle_oom_disabled(self, disabled_manager):
        """Test that disabled manager returns None."""
        new_size = disabled_manager.handle_oom(256)
        assert new_size is None

    def test_mark_success_resets_retries(self, manager):
        """Test that mark_success resets retry counter."""
        manager.handle_oom(256)
        manager.handle_oom(128)
        assert manager._current_retries == 2

        manager.mark_success()
        assert manager._current_retries == 0

    def test_reset(self, manager):
        """Test reset method."""
        manager.handle_oom(256)
        manager.handle_oom(128)
        assert manager._current_retries == 2

        manager.reset()
        assert manager._current_retries == 0

    def test_events_tracking(self, manager):
        """Test that OOM events are tracked."""
        assert manager.total_oom_count == 0

        manager.handle_oom(256)
        assert manager.total_oom_count == 1

        manager.handle_oom(128)
        assert manager.total_oom_count == 2

        events = manager.events
        assert len(events) == 2
        assert events[0].original_batch_size == 256
        assert events[0].reduced_batch_size == 128
        assert events[1].original_batch_size == 128
        assert events[1].reduced_batch_size == 64

    def test_recovered_count(self, manager):
        """Test recovered event counting."""
        manager.handle_oom(256)
        manager.handle_oom(128)
        assert manager.recovered_count == 0

        manager.mark_success()
        assert manager.recovered_count == 1

    def test_oom_safe_context(self, manager):
        """Test oom_safe_context context manager."""
        with manager.oom_safe_context() as ctx:
            assert not ctx.should_stop
            ctx.record_retry()
            ctx.record_retry()
            assert ctx.retries == 2
            ctx.mark_success()

        assert manager._current_retries == 0  # Reset after success

    def test_get_summary(self, manager):
        """Test get_summary method."""
        manager.handle_oom(256)
        manager.mark_success()

        summary = manager.get_summary()
        assert summary["enabled"] is True
        assert summary["total_oom_events"] == 1
        assert summary["recovered_events"] == 1
        assert summary["failed_events"] == 0
        assert len(summary["events"]) == 1

    def test_custom_reduction_factor(self):
        """Test custom batch reduction factor."""
        config = OOMConfig(batch_reduction_factor=0.75)
        manager = OOMRecoveryManager(config)

        new_size = manager.handle_oom(100)
        assert new_size == 75  # 100 * 0.75


class TestCreateOOMManager:
    """Tests for create_oom_manager factory function."""

    def test_create_with_defaults(self):
        """Test creating manager with defaults."""
        manager = create_oom_manager()
        assert manager.config.enabled is True
        assert manager.config.max_retries == 3

    def test_create_with_custom_values(self):
        """Test creating manager with custom values."""
        manager = create_oom_manager(
            enabled=False,
            max_retries=5,
            batch_reduction_factor=0.6,
            min_batch_size=16,
        )
        assert manager.config.enabled is False
        assert manager.config.max_retries == 5
        assert manager.config.batch_reduction_factor == 0.6
        assert manager.config.min_batch_size == 16


class TestOOMEvent:
    """Tests for OOMEvent dataclass."""

    def test_event_creation(self):
        """Test OOMEvent creation."""
        event = OOMEvent(
            original_batch_size=256,
            reduced_batch_size=128,
            retry_number=1,
            recovered=True,
        )
        assert event.original_batch_size == 256
        assert event.reduced_batch_size == 128
        assert event.retry_number == 1
        assert event.recovered is True
        assert event.memory_allocated is None
