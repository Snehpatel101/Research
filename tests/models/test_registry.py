"""
Tests for ModelRegistry - Plugin system for model types.

Tests cover:
- Model registration with decorator
- Model creation via registry
- Model discovery (list_models, get_model_info)
- Alias resolution
- Unknown model error handling
- Registry management (clear, count)
"""
import pytest

from src.models.base import BaseModel, PredictionOutput, TrainingMetrics
from src.models.registry import ModelRegistry, register


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def isolate_registry():
    """
    Store and restore registry state around each test.

    This ensures tests don't pollute each other with registered models.
    """
    # Store current state
    original_models = ModelRegistry._models.copy()
    original_families = {k: list(v) for k, v in ModelRegistry._families.items()}
    original_metadata = {k: v.copy() for k, v in ModelRegistry._metadata.items()}

    yield

    # Restore original state
    ModelRegistry._models = original_models
    ModelRegistry._families = {k: list(v) for k, v in original_families.items()}
    ModelRegistry._metadata = {k: v.copy() for k, v in original_metadata.items()}


@pytest.fixture
def mock_model_class():
    """Create a mock model class for testing registration."""
    class MockModel(BaseModel):
        @property
        def model_family(self) -> str:
            return "test"

        @property
        def requires_scaling(self) -> bool:
            return False

        @property
        def requires_sequences(self) -> bool:
            return False

        def get_default_config(self):
            return {"param": 1}

        def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
            self._is_fitted = True
            import numpy as np
            return TrainingMetrics(
                train_loss=0.1, val_loss=0.2,
                train_accuracy=0.9, val_accuracy=0.85,
                train_f1=0.88, val_f1=0.82,
                epochs_trained=10, training_time_seconds=1.0,
                early_stopped=False, best_epoch=10
            )

        def predict(self, X):
            import numpy as np
            n = len(X) if hasattr(X, '__len__') else X.shape[0]
            return PredictionOutput(
                class_predictions=np.zeros(n, dtype=int),
                class_probabilities=np.ones((n, 3)) / 3,
                confidence=np.ones(n) / 3
            )

        def save(self, path):
            pass

        def load(self, path):
            self._is_fitted = True

    return MockModel


# =============================================================================
# TEST MODEL REGISTRATION
# =============================================================================

class TestModelRegistration:
    """Tests for model registration using the decorator."""

    def test_register_basic(self, mock_model_class):
        """Should register a model with name and family."""
        @ModelRegistry.register("test_model", family="test_family")
        class TestModel(mock_model_class):
            pass

        assert ModelRegistry.is_registered("test_model")
        assert "test_family" in ModelRegistry.families()
        assert "test_model" in ModelRegistry.list_family("test_family")

    def test_register_with_description(self, mock_model_class):
        """Should store model description in metadata."""
        @ModelRegistry.register(
            "described_model",
            family="test",
            description="A test model with description"
        )
        class DescribedModel(mock_model_class):
            pass

        metadata = ModelRegistry.get_metadata("described_model")
        assert metadata["description"] == "A test model with description"

    def test_register_with_aliases(self, mock_model_class):
        """Should register model aliases."""
        @ModelRegistry.register(
            "aliased_model",
            family="test",
            aliases=["alias1", "alias2"]
        )
        class AliasedModel(mock_model_class):
            pass

        assert ModelRegistry.is_registered("aliased_model")
        assert ModelRegistry.is_registered("alias1")
        assert ModelRegistry.is_registered("alias2")

        # All should resolve to same class
        assert ModelRegistry.get("alias1") == ModelRegistry.get("aliased_model")
        assert ModelRegistry.get("alias2") == ModelRegistry.get("aliased_model")

    def test_register_duplicate_raises(self, mock_model_class):
        """Should raise ValueError when registering duplicate name."""
        @ModelRegistry.register("duplicate_model", family="test")
        class FirstModel(mock_model_class):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @ModelRegistry.register("duplicate_model", family="test")
            class SecondModel(mock_model_class):
                pass

    def test_register_non_basemodel_raises(self):
        """Should raise TypeError when registering non-BaseModel class."""
        with pytest.raises(TypeError, match="must be a subclass of BaseModel"):
            @ModelRegistry.register("invalid_model", family="test")
            class NotAModel:
                pass

    def test_convenience_register_function(self, mock_model_class):
        """Should work with convenience register function."""
        @register("convenience_model", family="convenience")
        class ConvenienceModel(mock_model_class):
            pass

        assert ModelRegistry.is_registered("convenience_model")


# =============================================================================
# TEST MODEL CREATION
# =============================================================================

class TestModelCreation:
    """Tests for creating models via registry."""

    def test_create_basic(self, mock_model_class):
        """Should create model instance by name."""
        @ModelRegistry.register("creatable_model", family="test")
        class CreatableModel(mock_model_class):
            pass

        model = ModelRegistry.create("creatable_model")
        assert isinstance(model, CreatableModel)

    def test_create_with_config(self, mock_model_class):
        """Should pass config to model constructor."""
        @ModelRegistry.register("configurable_model", family="test")
        class ConfigurableModel(mock_model_class):
            pass

        config = {"custom_param": 42}
        model = ModelRegistry.create("configurable_model", config=config)
        assert model.config.get("custom_param") == 42

    def test_create_via_alias(self, mock_model_class):
        """Should create model via alias."""
        @ModelRegistry.register("main_model", family="test", aliases=["shortname"])
        class MainModel(mock_model_class):
            pass

        model = ModelRegistry.create("shortname")
        assert isinstance(model, MainModel)

    def test_create_case_insensitive(self, mock_model_class):
        """Should handle case-insensitive model names."""
        @ModelRegistry.register("mixedcase", family="test")
        class MixedCaseModel(mock_model_class):
            pass

        model1 = ModelRegistry.create("MixedCase")
        model2 = ModelRegistry.create("MIXEDCASE")
        model3 = ModelRegistry.create("mixedcase")

        assert type(model1) == type(model2) == type(model3)

    def test_create_strips_whitespace(self, mock_model_class):
        """Should strip whitespace from model names."""
        @ModelRegistry.register("whitespace_model", family="test")
        class WhitespaceModel(mock_model_class):
            pass

        model = ModelRegistry.create("  whitespace_model  ")
        assert isinstance(model, WhitespaceModel)

    def test_create_unknown_raises(self):
        """Should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.create("nonexistent_model")


# =============================================================================
# TEST MODEL DISCOVERY
# =============================================================================

class TestModelDiscovery:
    """Tests for discovering registered models."""

    def test_list_models(self, mock_model_class):
        """Should list models grouped by family."""
        @ModelRegistry.register("model_a", family="family_1")
        class ModelA(mock_model_class):
            pass

        @ModelRegistry.register("model_b", family="family_1")
        class ModelB(mock_model_class):
            pass

        @ModelRegistry.register("model_c", family="family_2")
        class ModelC(mock_model_class):
            pass

        models = ModelRegistry.list_models()
        assert "family_1" in models
        assert "family_2" in models
        assert "model_a" in models["family_1"]
        assert "model_b" in models["family_1"]
        assert "model_c" in models["family_2"]

    def test_list_all(self, mock_model_class):
        """Should list all model names (excluding aliases)."""
        @ModelRegistry.register("all_model_1", family="test", aliases=["alias_1"])
        class Model1(mock_model_class):
            pass

        @ModelRegistry.register("all_model_2", family="test")
        class Model2(mock_model_class):
            pass

        all_models = ModelRegistry.list_all()
        assert "all_model_1" in all_models
        assert "all_model_2" in all_models
        # Aliases should not be in list_all
        assert "alias_1" not in all_models

    def test_list_family(self, mock_model_class):
        """Should list models in specific family."""
        @ModelRegistry.register("fam_model_1", family="specific_family")
        class FamModel1(mock_model_class):
            pass

        @ModelRegistry.register("fam_model_2", family="specific_family")
        class FamModel2(mock_model_class):
            pass

        @ModelRegistry.register("other_model", family="other_family")
        class OtherModel(mock_model_class):
            pass

        family_models = ModelRegistry.list_family("specific_family")
        assert "fam_model_1" in family_models
        assert "fam_model_2" in family_models
        assert "other_model" not in family_models

    def test_list_family_unknown_raises(self):
        """Should raise ValueError for unknown family."""
        with pytest.raises(ValueError, match="Unknown family"):
            ModelRegistry.list_family("nonexistent_family")

    def test_families(self, mock_model_class):
        """Should return list of all families."""
        @ModelRegistry.register("fam_test_1", family="test_fam_a")
        class TestA(mock_model_class):
            pass

        @ModelRegistry.register("fam_test_2", family="test_fam_b")
        class TestB(mock_model_class):
            pass

        families = ModelRegistry.families()
        assert "test_fam_a" in families
        assert "test_fam_b" in families


# =============================================================================
# TEST MODEL INFO AND METADATA
# =============================================================================

class TestModelInfo:
    """Tests for getting model information."""

    def test_get_metadata(self, mock_model_class):
        """Should return model metadata."""
        @ModelRegistry.register(
            "metadata_model",
            family="meta_family",
            description="Test description",
            aliases=["meta_alias"]
        )
        class MetadataModel(mock_model_class):
            pass

        metadata = ModelRegistry.get_metadata("metadata_model")
        assert metadata["name"] == "metadata_model"
        assert metadata["family"] == "meta_family"
        assert metadata["description"] == "Test description"
        assert "meta_alias" in metadata["aliases"]
        assert metadata["class"] == "MetadataModel"

    def test_get_metadata_via_alias(self, mock_model_class):
        """Should get metadata when using alias."""
        @ModelRegistry.register(
            "original_name",
            family="test",
            aliases=["alias_name"]
        )
        class OriginalModel(mock_model_class):
            pass

        metadata = ModelRegistry.get_metadata("alias_name")
        assert metadata["name"] == "original_name"

    def test_get_metadata_unknown_raises(self):
        """Should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get_metadata("nonexistent")

    def test_get_model_info(self, mock_model_class):
        """Should return detailed model info including runtime properties."""
        @ModelRegistry.register("info_model", family="test")
        class InfoModel(mock_model_class):
            @property
            def requires_scaling(self):
                return True

            @property
            def requires_sequences(self):
                return True

            def get_default_config(self):
                return {"test_param": 123}

        info = ModelRegistry.get_model_info("info_model")
        assert info["name"] == "info_model"
        assert info["requires_scaling"] is True
        assert info["requires_sequences"] is True
        assert info["default_config"]["test_param"] == 123

    def test_get_model_info_unknown_raises(self):
        """Should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get_model_info("nonexistent")


# =============================================================================
# TEST REGISTRY MANAGEMENT
# =============================================================================

class TestRegistryManagement:
    """Tests for registry management functions."""

    def test_is_registered(self, mock_model_class):
        """Should check if model is registered."""
        assert not ModelRegistry.is_registered("not_registered")

        @ModelRegistry.register("is_registered_model", family="test")
        class RegisteredModel(mock_model_class):
            pass

        assert ModelRegistry.is_registered("is_registered_model")

    def test_count(self, mock_model_class):
        """Should return count of registered models."""
        initial_count = ModelRegistry.count()

        @ModelRegistry.register("count_model_1", family="test")
        class CountModel1(mock_model_class):
            pass

        @ModelRegistry.register("count_model_2", family="test", aliases=["alias"])
        class CountModel2(mock_model_class):
            pass

        # Count should increase by 2 (aliases don't count)
        assert ModelRegistry.count() == initial_count + 2

    def test_get_class(self, mock_model_class):
        """Should return model class (not instance)."""
        @ModelRegistry.register("get_class_model", family="test")
        class GetClassModel(mock_model_class):
            pass

        model_class = ModelRegistry.get("get_class_model")
        assert model_class is GetClassModel
        assert not isinstance(model_class, BaseModel)  # It's a class, not instance

    def test_clear(self, mock_model_class):
        """Should clear all registered models."""
        @ModelRegistry.register("clear_model", family="test")
        class ClearModel(mock_model_class):
            pass

        assert ModelRegistry.is_registered("clear_model")

        ModelRegistry.clear()

        assert not ModelRegistry.is_registered("clear_model")
        assert ModelRegistry.count() == 0
        assert len(ModelRegistry.families()) == 0


# =============================================================================
# TEST INTEGRATION WITH REAL MODELS
# =============================================================================

class TestIntegrationWithRealModels:
    """Integration tests with actual registered models."""

    def test_xgboost_is_registered(self):
        """XGBoost should be registered in boosting family."""
        # Import to trigger registration
        from src.models.boosting import XGBoostModel

        assert ModelRegistry.is_registered("xgboost")
        assert ModelRegistry.is_registered("xgb")  # alias

        families = ModelRegistry.list_models()
        assert "boosting" in families
        assert "xgboost" in families["boosting"]

    def test_lightgbm_is_registered(self):
        """LightGBM should be registered if available."""
        try:
            from src.models.boosting import LightGBMModel
            assert ModelRegistry.is_registered("lightgbm")
            assert ModelRegistry.is_registered("lgbm")  # alias
        except ImportError:
            pytest.skip("LightGBM not installed")

    def test_catboost_is_registered(self):
        """CatBoost should be registered if available."""
        try:
            from src.models.boosting import CatBoostModel
            assert ModelRegistry.is_registered("catboost")
            assert ModelRegistry.is_registered("cat")  # alias
        except ImportError:
            pytest.skip("CatBoost not installed")

    def test_neural_models_registered(self):
        """Neural models should be registered if PyTorch available."""
        try:
            from src.models.neural import LSTMModel, GRUModel, TCNModel

            assert ModelRegistry.is_registered("lstm")
            assert ModelRegistry.is_registered("gru")
            assert ModelRegistry.is_registered("tcn")

            families = ModelRegistry.list_models()
            assert "neural" in families
            assert "lstm" in families["neural"]
            assert "gru" in families["neural"]
            assert "tcn" in families["neural"]
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_create_real_xgboost(self):
        """Should create real XGBoost model via registry."""
        from src.models.boosting import XGBoostModel

        model = ModelRegistry.create("xgboost", config={"n_estimators": 10})
        assert isinstance(model, XGBoostModel)
        assert model.config["n_estimators"] == 10

    def test_model_info_xgboost(self):
        """Should get correct info for XGBoost."""
        from src.models.boosting import XGBoostModel

        info = ModelRegistry.get_model_info("xgboost")
        assert info["family"] == "boosting"
        assert info["requires_scaling"] is False
        assert info["requires_sequences"] is False
        assert "n_estimators" in info["default_config"]
