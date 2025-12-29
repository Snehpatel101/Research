"""
Tests for LazyImports singleton pattern.

Covers:
1. Singleton behavior (only one instance created)
2. Lazy loading of modules
3. Module caching after first access
4. All properties work correctly
5. Backward compatibility with deprecated functions
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.cli.run_commands_core import (
    LazyImports,
    _get_pipeline_config,
    _get_pipeline_runner,
    _get_presets_module,
    _get_model_config,
)


# =============================================================================
# SINGLETON BEHAVIOR TESTS
# =============================================================================


class TestSingletonBehavior:
    """Tests for singleton pattern implementation."""

    def test_singleton_returns_same_instance(self) -> None:
        """Test that LazyImports always returns the same instance."""
        instance1 = LazyImports()
        instance2 = LazyImports()
        instance3 = LazyImports()

        assert instance1 is instance2
        assert instance2 is instance3
        assert instance1 is instance3

    def test_singleton_identity(self) -> None:
        """Test that singleton instance has consistent identity."""
        instance = LazyImports()

        # Create new references
        ref1 = LazyImports()
        ref2 = LazyImports()

        # All should be the exact same object
        assert id(instance) == id(ref1) == id(ref2)

    def test_singleton_state_persists(self) -> None:
        """Test that singleton state persists across multiple instantiations."""
        # Force load one module
        instance1 = LazyImports()
        _ = instance1.pipeline_config  # Load module

        # Get new reference
        instance2 = LazyImports()

        # Both should have the module loaded
        assert instance1._pipeline_config is not None
        assert instance2._pipeline_config is not None
        assert instance1._pipeline_config is instance2._pipeline_config


# =============================================================================
# LAZY LOADING TESTS
# =============================================================================


class TestLazyLoading:
    """Tests for lazy loading behavior."""

    def test_modules_not_loaded_initially(self) -> None:
        """Test that modules are not loaded on instantiation."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()

        # All private attributes should be None
        assert instance._pipeline_config is None
        assert instance._pipeline_runner is None
        assert instance._presets_module is None
        assert instance._model_config is None
        assert instance._manifest is None

    def test_pipeline_config_lazy_loads(self) -> None:
        """Test that pipeline_config loads on first access."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()
        assert instance._pipeline_config is None

        # Access property - should trigger load
        config = instance.pipeline_config

        assert config is not None
        assert instance._pipeline_config is not None

    def test_pipeline_runner_lazy_loads(self) -> None:
        """Test that pipeline_runner loads on first access."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()
        assert instance._pipeline_runner is None

        # Access property - should trigger load
        runner = instance.pipeline_runner

        assert runner is not None
        assert instance._pipeline_runner is not None

    def test_presets_lazy_loads(self) -> None:
        """Test that presets loads on first access."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()
        assert instance._presets_module is None

        # Access property - should trigger load
        presets = instance.presets

        assert presets is not None
        assert instance._presets_module is not None

    def test_model_config_lazy_loads(self) -> None:
        """Test that model_config loads on first access."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()
        assert instance._model_config is None

        # Access property - should trigger load
        config = instance.model_config

        assert config is not None
        assert instance._model_config is not None

    def test_manifest_lazy_loads(self) -> None:
        """Test that manifest loads on first access."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()
        assert instance._manifest is None

        # Access property - should trigger load
        manifest = instance.manifest

        assert manifest is not None
        assert instance._manifest is not None


# =============================================================================
# MODULE CACHING TESTS
# =============================================================================


class TestModuleCaching:
    """Tests for module caching behavior."""

    def test_pipeline_config_cached_after_first_access(self) -> None:
        """Test that pipeline_config is cached after first access."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()

        # First access
        config1 = instance.pipeline_config
        cached1 = instance._pipeline_config

        # Second access - should return cached module
        config2 = instance.pipeline_config
        cached2 = instance._pipeline_config

        assert config1 is config2
        assert cached1 is cached2

    def test_all_modules_cache_independently(self) -> None:
        """Test that each module is cached independently."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        instance = LazyImports()

        # Load all modules
        pipeline_config = instance.pipeline_config
        pipeline_runner = instance.pipeline_runner
        presets = instance.presets
        model_config = instance.model_config
        manifest = instance.manifest

        # All should be cached
        assert instance._pipeline_config is pipeline_config
        assert instance._pipeline_runner is pipeline_runner
        assert instance._presets_module is presets
        assert instance._model_config is model_config
        assert instance._manifest is manifest

        # Access again - should return same cached instances
        assert instance.pipeline_config is pipeline_config
        assert instance.pipeline_runner is pipeline_runner
        assert instance.presets is presets
        assert instance.model_config is model_config
        assert instance.manifest is manifest


# =============================================================================
# PROPERTY FUNCTIONALITY TESTS
# =============================================================================


class TestPropertyFunctionality:
    """Tests that each property works correctly."""

    def test_pipeline_config_property_returns_module(self) -> None:
        """Test that pipeline_config property returns a module."""
        instance = LazyImports()
        config = instance.pipeline_config

        # Should be a module with expected attributes
        assert hasattr(config, 'PipelineConfig')
        assert hasattr(config, 'create_default_config')

    def test_pipeline_runner_property_returns_module(self) -> None:
        """Test that pipeline_runner property returns a module."""
        instance = LazyImports()
        runner = instance.pipeline_runner

        # Should be a module
        assert runner is not None
        # Note: Can't check for PipelineRunner attribute without torch installed
        # The important thing is that the module loads without error when dependencies are met

    def test_presets_property_returns_module(self) -> None:
        """Test that presets property returns a module."""
        instance = LazyImports()
        presets = instance.presets

        # Should be a module with expected attributes
        assert hasattr(presets, 'validate_preset')
        assert hasattr(presets, 'get_preset')
        assert hasattr(presets, 'list_available_presets')

    def test_model_config_property_returns_module(self) -> None:
        """Test that model_config property returns a module."""
        instance = LazyImports()
        config = instance.model_config

        # Should be a module with expected attributes
        assert hasattr(config, 'get_all_model_names')
        assert hasattr(config, 'get_model_requirements')

    def test_manifest_property_returns_module(self) -> None:
        """Test that manifest property returns a module."""
        instance = LazyImports()
        manifest = instance.manifest

        # Should be a module with expected attributes
        assert hasattr(manifest, 'ArtifactManifest')


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with deprecated functions."""

    def test_get_pipeline_config_function_works(self) -> None:
        """Test that _get_pipeline_config() function still works."""
        config = _get_pipeline_config()

        assert config is not None
        assert hasattr(config, 'PipelineConfig')

    def test_get_pipeline_runner_function_works(self) -> None:
        """Test that _get_pipeline_runner() function still works."""
        runner = _get_pipeline_runner()

        assert runner is not None
        # Note: Can't check for PipelineRunner attribute without torch installed

    def test_get_presets_module_function_works(self) -> None:
        """Test that _get_presets_module() function still works."""
        presets = _get_presets_module()

        assert presets is not None
        assert hasattr(presets, 'validate_preset')

    def test_get_model_config_function_works(self) -> None:
        """Test that _get_model_config() function still works."""
        config = _get_model_config()

        assert config is not None
        assert hasattr(config, 'get_all_model_names')

    def test_backward_compat_functions_use_singleton(self) -> None:
        """Test that backward compat functions use the singleton."""
        # Get modules via deprecated functions
        config1 = _get_pipeline_config()
        runner1 = _get_pipeline_runner()
        presets1 = _get_presets_module()
        model_config1 = _get_model_config()

        # Get modules via singleton
        lazy = LazyImports()
        config2 = lazy.pipeline_config
        runner2 = lazy.pipeline_runner
        presets2 = lazy.presets
        model_config2 = lazy.model_config

        # Should be the same instances (from singleton cache)
        assert config1 is config2
        assert runner1 is runner2
        assert presets1 is presets2
        assert model_config1 is model_config2


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestInitialization:
    """Tests for initialization behavior."""

    def test_init_only_runs_once(self) -> None:
        """Test that __init__ only initializes state once."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        # First instantiation
        instance1 = LazyImports()
        assert LazyImports._initialized is True

        # Load a module
        _ = instance1.pipeline_config

        # Second instantiation should not reset state
        instance2 = LazyImports()

        # _initialized flag should still be True
        assert LazyImports._initialized is True

        # Module should still be cached
        assert instance2._pipeline_config is not None
        assert instance1._pipeline_config is instance2._pipeline_config

    def test_multiple_instantiations_preserve_loaded_modules(self) -> None:
        """Test that multiple instantiations don't reload modules."""
        # Reset singleton for clean test
        LazyImports._instance = None
        LazyImports._initialized = False

        # Load all modules
        instance1 = LazyImports()
        config = instance1.pipeline_config
        runner = instance1.pipeline_runner
        presets = instance1.presets

        # Create new references
        instance2 = LazyImports()
        instance3 = LazyImports()

        # All should have modules cached
        assert instance2._pipeline_config is config
        assert instance2._pipeline_runner is runner
        assert instance2._presets_module is presets

        assert instance3._pipeline_config is config
        assert instance3._pipeline_runner is runner
        assert instance3._presets_module is presets


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
