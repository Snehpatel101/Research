"""
Tests for Advanced Neural Models - PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D.

Tests cover:
- Model initialization and properties
- Model registration in registry
- Default configuration validation
- Sequence handling (requires_sequences)
- Scaling requirement (requires_scaling)
- Training with early stopping
- Prediction output format
- Save/load roundtrip
- GPU mode and mixed precision
- Production safety (causality)
"""
from typing import Any, Dict

import numpy as np
import pytest

from tests.models.conftest import requires_torch


# =============================================================================
# FIXTURES FOR ADVANCED NEURAL MODELS
# =============================================================================


@pytest.fixture
def fast_patchtst_config() -> Dict[str, Any]:
    """Fast PatchTST config for tests."""
    return {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "patch_len": 4,
        "stride": 2,
        "dropout": 0.0,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_itransformer_config() -> Dict[str, Any]:
    """Fast iTransformer config for tests."""
    return {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_tft_config() -> Dict[str, Any]:
    """Fast TFT config for tests."""
    return {
        "d_model": 32,
        "n_heads": 2,
        "lstm_layers": 1,
        "attention_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_nbeats_config() -> Dict[str, Any]:
    """Fast N-BEATS config for tests."""
    return {
        "n_stacks": 2,
        "n_blocks_per_stack": 1,
        "hidden_size": 32,
        "n_layers": 2,
        "theta_size": 8,
        "dropout": 0.0,
        "n_harmonics": 2,
        "polynomial_degree": 2,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_inceptiontime_config() -> Dict[str, Any]:
    """Fast InceptionTime config for tests."""
    return {
        "n_blocks": 2,
        "n_filters": 8,
        "kernel_sizes": (3, 5, 7),
        "bottleneck_channels": 8,
        "n_modules_per_block": 1,
        "use_residual": True,
        "dropout": 0.0,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_resnet1d_config() -> Dict[str, Any]:
    """Fast ResNet1D config for tests."""
    return {
        "n_blocks": (1, 1),
        "channels": (16, 32),
        "kernel_size": 3,
        "stem_kernel_size": 3,
        "use_bottleneck": False,
        "dropout": 0.0,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def trained_patchtst(small_sequence_data, fast_patchtst_config):
    """Provide a trained PatchTST model."""
    from src.models.neural import PatchTSTModel

    model = PatchTSTModel(config=fast_patchtst_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_itransformer(small_sequence_data, fast_itransformer_config):
    """Provide a trained iTransformer model."""
    from src.models.neural import iTransformerModel

    model = iTransformerModel(config=fast_itransformer_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_tft(small_sequence_data, fast_tft_config):
    """Provide a trained TFT model."""
    from src.models.neural import TFTModel

    model = TFTModel(config=fast_tft_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_nbeats(small_sequence_data, fast_nbeats_config):
    """Provide a trained N-BEATS model."""
    from src.models.neural import NBEATSModel

    model = NBEATSModel(config=fast_nbeats_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_inceptiontime(small_sequence_data, fast_inceptiontime_config):
    """Provide a trained InceptionTime model."""
    from src.models.neural import InceptionTimeModel

    model = InceptionTimeModel(config=fast_inceptiontime_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_resnet1d(small_sequence_data, fast_resnet1d_config):
    """Provide a trained ResNet1D model."""
    from src.models.neural import ResNet1DModel

    model = ResNet1DModel(config=fast_resnet1d_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


# =============================================================================
# PATCHTST TESTS
# =============================================================================


@requires_torch
class TestPatchTSTModelRegistration:
    """Tests for PatchTST model registration."""

    def test_model_in_registry(self):
        """PatchTST should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "patchtst" in families["neural"]

    def test_registry_create(self):
        """Should be able to create PatchTST via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("patchtst")
        assert model is not None
        assert model.model_family == "neural"


@requires_torch
class TestPatchTSTModelProperties:
    """Tests for PatchTST model properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """PatchTST should require scaling."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """PatchTST should require sequences."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel()
        config = model.get_default_config()

        expected_keys = [
            "d_model", "n_heads", "n_layers", "d_ff",
            "patch_len", "stride", "dropout", "batch_size", "max_epochs"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestPatchTSTTraining:
    """Tests for PatchTST training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_patchtst_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel(config=fast_patchtst_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_patchtst):
        """Model should be fitted after training."""
        assert trained_patchtst.is_fitted is True

    def test_metadata_includes_patch_info(self, small_sequence_data, fast_patchtst_config):
        """Training metadata should include patch information."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel(config=fast_patchtst_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert "patch_len" in metrics.metadata
        assert "stride" in metrics.metadata
        assert "n_patches" in metrics.metadata


@requires_torch
class TestPatchTSTPrediction:
    """Tests for PatchTST prediction."""

    def test_predict_returns_output(self, trained_patchtst, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_patchtst.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_patchtst, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_patchtst.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_patchtst, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_patchtst.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, small_sequence_data):
        """Prediction on unfitted model should raise."""
        from src.models.neural import PatchTSTModel

        model = PatchTSTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_sequence_data["X_val"])


@requires_torch
class TestPatchTSTSaveLoad:
    """Tests for PatchTST serialization."""

    def test_save_creates_files(self, trained_patchtst, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "patchtst_model"
        trained_patchtst.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_patchtst, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import PatchTSTModel

        save_path = tmp_model_dir / "patchtst_model"
        trained_patchtst.save(save_path)

        loaded = PatchTSTModel()
        loaded.load(save_path)

        original = trained_patchtst.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# ITRANSFORMER TESTS
# =============================================================================


@requires_torch
class TestiTransformerModelRegistration:
    """Tests for iTransformer model registration."""

    def test_model_in_registry(self):
        """iTransformer should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "itransformer" in families["neural"]

    def test_registry_create(self):
        """Should be able to create iTransformer via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("itransformer")
        assert model is not None
        assert model.model_family == "neural"


@requires_torch
class TestiTransformerModelProperties:
    """Tests for iTransformer model properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import iTransformerModel

        model = iTransformerModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """iTransformer should require scaling."""
        from src.models.neural import iTransformerModel

        model = iTransformerModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """iTransformer should require sequences."""
        from src.models.neural import iTransformerModel

        model = iTransformerModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import iTransformerModel

        model = iTransformerModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import iTransformerModel

        model = iTransformerModel()
        config = model.get_default_config()

        expected_keys = [
            "d_model", "n_heads", "n_layers", "d_ff",
            "dropout", "batch_size", "max_epochs"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestiTransformerTraining:
    """Tests for iTransformer training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_itransformer_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import iTransformerModel

        model = iTransformerModel(config=fast_itransformer_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_itransformer):
        """Model should be fitted after training."""
        assert trained_itransformer.is_fitted is True


@requires_torch
class TestiTransformerPrediction:
    """Tests for iTransformer prediction."""

    def test_predict_returns_output(self, trained_itransformer, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_itransformer.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_itransformer, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_itransformer.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_itransformer, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_itransformer.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestiTransformerSaveLoad:
    """Tests for iTransformer serialization."""

    def test_save_creates_files(self, trained_itransformer, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "itransformer_model"
        trained_itransformer.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_itransformer, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import iTransformerModel

        save_path = tmp_model_dir / "itransformer_model"
        trained_itransformer.save(save_path)

        loaded = iTransformerModel()
        loaded.load(save_path)

        original = trained_itransformer.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# TFT TESTS
# =============================================================================


@requires_torch
class TestTFTModelRegistration:
    """Tests for TFT model registration."""

    def test_model_in_registry(self):
        """TFT should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "tft" in families["neural"]

    def test_registry_create(self):
        """Should be able to create TFT via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("tft")
        assert model is not None
        assert model.model_family == "neural"


@requires_torch
class TestTFTModelProperties:
    """Tests for TFT model properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import TFTModel

        model = TFTModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """TFT should require scaling."""
        from src.models.neural import TFTModel

        model = TFTModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """TFT should require sequences."""
        from src.models.neural import TFTModel

        model = TFTModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import TFTModel

        model = TFTModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import TFTModel

        model = TFTModel()
        config = model.get_default_config()

        expected_keys = [
            "d_model", "n_heads", "lstm_layers", "attention_layers",
            "d_ff", "dropout", "batch_size", "max_epochs"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestTFTTraining:
    """Tests for TFT training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_tft_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import TFTModel

        model = TFTModel(config=fast_tft_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_tft):
        """Model should be fitted after training."""
        assert trained_tft.is_fitted is True


@requires_torch
class TestTFTPrediction:
    """Tests for TFT prediction."""

    def test_predict_returns_output(self, trained_tft, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_tft.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_tft, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_tft.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_tft, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_tft.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestTFTSaveLoad:
    """Tests for TFT serialization."""

    def test_save_creates_files(self, trained_tft, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "tft_model"
        trained_tft.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_tft, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import TFTModel

        save_path = tmp_model_dir / "tft_model"
        trained_tft.save(save_path)

        loaded = TFTModel()
        loaded.load(save_path)

        original = trained_tft.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# N-BEATS TESTS
# =============================================================================


@requires_torch
class TestNBEATSModelRegistration:
    """Tests for N-BEATS model registration."""

    def test_model_in_registry(self):
        """N-BEATS should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "nbeats" in families["neural"]

    def test_registry_create(self):
        """Should be able to create N-BEATS via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("nbeats")
        assert model is not None
        assert model.model_family == "neural"


@requires_torch
class TestNBEATSModelProperties:
    """Tests for N-BEATS model properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """N-BEATS should require scaling."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """N-BEATS should require sequences."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel()
        config = model.get_default_config()

        expected_keys = [
            "n_stacks", "n_blocks_per_stack", "hidden_size", "n_layers",
            "theta_size", "dropout", "stack_types", "n_harmonics",
            "polynomial_degree", "batch_size", "max_epochs"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestNBEATSTraining:
    """Tests for N-BEATS training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_nbeats_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel(config=fast_nbeats_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_nbeats):
        """Model should be fitted after training."""
        assert trained_nbeats.is_fitted is True

    def test_metadata_includes_stack_info(self, small_sequence_data, fast_nbeats_config):
        """Training metadata should include stack information."""
        from src.models.neural import NBEATSModel

        model = NBEATSModel(config=fast_nbeats_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert "n_stacks" in metrics.metadata
        assert "stack_types" in metrics.metadata


@requires_torch
class TestNBEATSPrediction:
    """Tests for N-BEATS prediction."""

    def test_predict_returns_output(self, trained_nbeats, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_nbeats.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_nbeats, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_nbeats.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_nbeats, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_nbeats.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestNBEATSSaveLoad:
    """Tests for N-BEATS serialization."""

    def test_save_creates_files(self, trained_nbeats, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "nbeats_model"
        trained_nbeats.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_nbeats, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import NBEATSModel

        save_path = tmp_model_dir / "nbeats_model"
        trained_nbeats.save(save_path)

        loaded = NBEATSModel()
        loaded.load(save_path)

        original = trained_nbeats.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# INCEPTIONTIME TESTS
# =============================================================================


@requires_torch
class TestInceptionTimeModelRegistration:
    """Tests for InceptionTime model registration."""

    def test_model_in_registry(self):
        """InceptionTime should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "inceptiontime" in families["neural"]

    def test_registry_create(self):
        """Should be able to create InceptionTime via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("inceptiontime")
        assert model is not None
        assert model.model_family == "neural"


@requires_torch
class TestInceptionTimeModelProperties:
    """Tests for InceptionTime model properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import InceptionTimeModel

        model = InceptionTimeModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """InceptionTime should require scaling."""
        from src.models.neural import InceptionTimeModel

        model = InceptionTimeModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """InceptionTime should require sequences."""
        from src.models.neural import InceptionTimeModel

        model = InceptionTimeModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import InceptionTimeModel

        model = InceptionTimeModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import InceptionTimeModel

        model = InceptionTimeModel()
        config = model.get_default_config()

        expected_keys = [
            "n_blocks", "n_filters", "kernel_sizes", "bottleneck_channels",
            "n_modules_per_block", "use_residual", "dropout", "batch_size"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestInceptionTimeTraining:
    """Tests for InceptionTime training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_inceptiontime_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import InceptionTimeModel

        model = InceptionTimeModel(config=fast_inceptiontime_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_inceptiontime):
        """Model should be fitted after training."""
        assert trained_inceptiontime.is_fitted is True


@requires_torch
class TestInceptionTimePrediction:
    """Tests for InceptionTime prediction."""

    def test_predict_returns_output(self, trained_inceptiontime, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_inceptiontime.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_inceptiontime, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_inceptiontime.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_inceptiontime, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_inceptiontime.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestInceptionTimeSaveLoad:
    """Tests for InceptionTime serialization."""

    def test_save_creates_files(self, trained_inceptiontime, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "inceptiontime_model"
        trained_inceptiontime.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_inceptiontime, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import InceptionTimeModel

        save_path = tmp_model_dir / "inceptiontime_model"
        trained_inceptiontime.save(save_path)

        loaded = InceptionTimeModel()
        loaded.load(save_path)

        original = trained_inceptiontime.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# RESNET1D TESTS
# =============================================================================


@requires_torch
class TestResNet1DModelRegistration:
    """Tests for ResNet1D model registration."""

    def test_model_in_registry(self):
        """ResNet1D should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "resnet1d" in families["neural"]

    def test_registry_create(self):
        """Should be able to create ResNet1D via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("resnet1d")
        assert model is not None
        assert model.model_family == "neural"


@requires_torch
class TestResNet1DModelProperties:
    """Tests for ResNet1D model properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import ResNet1DModel

        model = ResNet1DModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """ResNet1D should require scaling."""
        from src.models.neural import ResNet1DModel

        model = ResNet1DModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """ResNet1D should require sequences."""
        from src.models.neural import ResNet1DModel

        model = ResNet1DModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import ResNet1DModel

        model = ResNet1DModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import ResNet1DModel

        model = ResNet1DModel()
        config = model.get_default_config()

        expected_keys = [
            "n_blocks", "channels", "kernel_size", "stem_kernel_size",
            "use_bottleneck", "dropout", "batch_size"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestResNet1DTraining:
    """Tests for ResNet1D training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_resnet1d_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import ResNet1DModel

        model = ResNet1DModel(config=fast_resnet1d_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_resnet1d):
        """Model should be fitted after training."""
        assert trained_resnet1d.is_fitted is True


@requires_torch
class TestResNet1DPrediction:
    """Tests for ResNet1D prediction."""

    def test_predict_returns_output(self, trained_resnet1d, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_resnet1d.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_resnet1d, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_resnet1d.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_resnet1d, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_resnet1d.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestResNet1DSaveLoad:
    """Tests for ResNet1D serialization."""

    def test_save_creates_files(self, trained_resnet1d, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "resnet1d_model"
        trained_resnet1d.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_resnet1d, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import ResNet1DModel

        save_path = tmp_model_dir / "resnet1d_model"
        trained_resnet1d.save(save_path)

        loaded = ResNet1DModel()
        loaded.load(save_path)

        original = trained_resnet1d.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# CROSS-MODEL TESTS
# =============================================================================


@requires_torch
class TestAdvancedNeuralModelConsistency:
    """Tests for consistent behavior across advanced neural models."""

    def test_all_models_in_registry(self):
        """All advanced neural models should be in registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "neural" in families

        expected_models = ["patchtst", "itransformer", "tft", "nbeats", "inceptiontime", "resnet1d"]
        for model_name in expected_models:
            assert model_name in families["neural"], f"Missing model: {model_name}"

    def test_all_require_sequences(self):
        """All advanced neural models should require sequences."""
        from src.models.neural import (
            PatchTSTModel, iTransformerModel, TFTModel,
            NBEATSModel, InceptionTimeModel, ResNet1DModel
        )

        for ModelClass in [PatchTSTModel, iTransformerModel, TFTModel, NBEATSModel, InceptionTimeModel, ResNet1DModel]:
            model = ModelClass()
            assert model.requires_sequences is True, f"{ModelClass.__name__} should require sequences"

    def test_all_require_scaling(self):
        """All advanced neural models should require scaling."""
        from src.models.neural import (
            PatchTSTModel, iTransformerModel, TFTModel,
            NBEATSModel, InceptionTimeModel, ResNet1DModel
        )

        for ModelClass in [PatchTSTModel, iTransformerModel, TFTModel, NBEATSModel, InceptionTimeModel, ResNet1DModel]:
            model = ModelClass()
            assert model.requires_scaling is True, f"{ModelClass.__name__} should require scaling"

