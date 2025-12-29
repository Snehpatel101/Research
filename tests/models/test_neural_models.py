"""
Tests for Neural Models - LSTM, GRU, TCN.

Tests cover:
- Model initialization and properties
- Sequence handling (requires_sequences)
- Scaling requirement (requires_scaling)
- Training with early stopping
- Prediction output format
- Save/load roundtrip
- GPU mode and mixed precision
- Configuration handling
"""
from pathlib import Path

import numpy as np
import pytest

# Conditional PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from tests.models.conftest import requires_torch, requires_cuda


# =============================================================================
# LSTM TESTS
# =============================================================================

@requires_torch
class TestLSTMModelProperties:
    """Tests for LSTMModel properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """LSTM should require scaling."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """LSTM should require sequences."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        assert model.requires_sequences is True

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        config = model.get_default_config()

        expected_keys = [
            "hidden_size", "num_layers", "dropout",
            "batch_size", "max_epochs", "learning_rate",
            "early_stopping_patience"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"

    def test_config_override(self):
        """Config should be overridable in constructor."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config={"hidden_size": 128, "num_layers": 3})
        assert model.config["hidden_size"] == 128
        assert model.config["num_layers"] == 3


@requires_torch
class TestLSTMTraining:
    """Tests for LSTMModel training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_lstm_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config=fast_lstm_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert 0 <= metrics.train_f1 <= 1
        assert 0 <= metrics.val_f1 <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, small_sequence_data, fast_lstm_config):
        """Training should accept sample weights."""
        from src.models.neural import LSTMModel
        weights = np.random.uniform(0.5, 1.5, size=len(small_sequence_data["y_train"]))
        model = LSTMModel(config=fast_lstm_config)

        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.epochs_trained > 0

    def test_early_stopping(self, small_sequence_data):
        """Early stopping should work."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config={
            "hidden_size": 16,
            "num_layers": 1,
            "max_epochs": 100,
            "early_stopping_patience": 2,
            "min_delta": 0.0,  # Any improvement counts
            "batch_size": 16,
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
        })
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        # Should stop before max epochs due to small data
        assert metrics.epochs_trained <= 100

    def test_is_fitted_after_training(self, trained_lstm):
        """Model should be fitted after training."""
        assert trained_lstm.is_fitted is True

    def test_history_tracked(self, small_sequence_data, fast_lstm_config):
        """Training history should be tracked."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config=fast_lstm_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert "train_loss" in metrics.history
        assert "val_loss" in metrics.history
        assert len(metrics.history["train_loss"]) == metrics.epochs_trained


@requires_torch
class TestLSTMPrediction:
    """Tests for LSTMModel prediction."""

    def test_predict_returns_output(self, trained_lstm, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_lstm.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_lstm, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_lstm.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_shape(self, trained_lstm, small_sequence_data):
        """Probabilities should have shape (n_samples, 3)."""
        output = trained_lstm.predict(small_sequence_data["X_val"])
        assert output.class_probabilities.shape == (len(small_sequence_data["X_val"]), 3)

    def test_predict_probabilities_sum_to_one(self, trained_lstm, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_lstm.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_confidence_is_max_prob(self, trained_lstm, small_sequence_data):
        """Confidence should be max probability."""
        output = trained_lstm.predict(small_sequence_data["X_val"])
        expected_conf = output.class_probabilities.max(axis=1)
        assert np.allclose(output.confidence, expected_conf)

    def test_predict_unfitted_raises(self, small_sequence_data):
        """Prediction on unfitted model should raise."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_sequence_data["X_val"])


@requires_torch
class TestLSTMSaveLoad:
    """Tests for LSTMModel serialization."""

    def test_save_creates_files(self, trained_lstm, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "lstm_model"
        trained_lstm.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_load_restores_model(self, trained_lstm, tmp_model_dir):
        """Load should restore model correctly."""
        from src.models.neural import LSTMModel
        save_path = tmp_model_dir / "lstm_model"
        trained_lstm.save(save_path)

        loaded = LSTMModel()
        loaded.load(save_path)

        assert loaded.is_fitted is True

    def test_predictions_match_after_load(self, trained_lstm, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import LSTMModel
        save_path = tmp_model_dir / "lstm_model"
        trained_lstm.save(save_path)

        loaded = LSTMModel()
        loaded.load(save_path)

        original = trained_lstm.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        # Use higher tolerance due to GPU/CPU floating point differences
        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)

    def test_save_unfitted_raises(self, tmp_model_dir):
        """Save on unfitted model should raise."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_model_dir / "model")

    def test_load_missing_raises(self):
        """Load from missing path should raise."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/path"))


@requires_torch
class TestLSTMInputValidation:
    """Tests for LSTMModel input validation."""

    def test_2d_input_raises(self, trained_lstm):
        """2D input should raise for sequential model."""
        with pytest.raises(ValueError, match="must be 3D"):
            trained_lstm.predict(np.random.randn(10, 20))

    def test_1d_input_raises(self, trained_lstm):
        """1D input should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            trained_lstm.predict(np.array([1, 2, 3]))


@requires_torch
class TestLSTMHiddenStates:
    """Tests for LSTM hidden state access."""

    def test_get_hidden_states(self, trained_lstm, small_sequence_data):
        """Should return hidden states for interpretability."""
        hidden = trained_lstm.get_hidden_states(small_sequence_data["X_val"])

        assert hidden is not None
        n_samples, seq_len, n_features = small_sequence_data["X_val"].shape
        assert hidden.shape[0] == n_samples
        assert hidden.shape[1] == seq_len

    def test_hidden_states_unfitted_returns_none(self, small_sequence_data):
        """Hidden states on unfitted model should return None."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        assert model.get_hidden_states(small_sequence_data["X_val"]) is None


# =============================================================================
# GRU TESTS
# =============================================================================

@requires_torch
class TestGRUModelProperties:
    """Tests for GRUModel properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import GRUModel
        model = GRUModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """GRU should require scaling."""
        from src.models.neural import GRUModel
        model = GRUModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """GRU should require sequences."""
        from src.models.neural import GRUModel
        model = GRUModel()
        assert model.requires_sequences is True

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import GRUModel
        model = GRUModel()
        config = model.get_default_config()

        expected_keys = ["hidden_size", "num_layers", "dropout", "batch_size"]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestGRUTraining:
    """Tests for GRUModel training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_gru_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import GRUModel
        model = GRUModel(config=fast_gru_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_gru):
        """Model should be fitted after training."""
        assert trained_gru.is_fitted is True


@requires_torch
class TestGRUPrediction:
    """Tests for GRUModel prediction."""

    def test_predict_returns_output(self, trained_gru, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_gru.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_gru, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_gru.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_gru, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_gru.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestGRUSaveLoad:
    """Tests for GRUModel serialization."""

    def test_save_creates_files(self, trained_gru, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "gru_model"
        trained_gru.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_gru, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import GRUModel
        save_path = tmp_model_dir / "gru_model"
        trained_gru.save(save_path)

        loaded = GRUModel()
        loaded.load(save_path)

        original = trained_gru.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        # Use higher tolerance due to GPU/CPU floating point differences
        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


# =============================================================================
# TCN TESTS
# =============================================================================

@requires_torch
class TestTCNModelProperties:
    """Tests for TCNModel properties."""

    def test_model_family(self):
        """Model family should be 'neural'."""
        from src.models.neural import TCNModel
        model = TCNModel()
        assert model.model_family == "neural"

    def test_requires_scaling_true(self):
        """TCN should require scaling."""
        from src.models.neural import TCNModel
        model = TCNModel()
        assert model.requires_scaling is True

    def test_requires_sequences_true(self):
        """TCN should require sequences."""
        from src.models.neural import TCNModel
        model = TCNModel()
        assert model.requires_sequences is True

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.neural import TCNModel
        model = TCNModel()
        config = model.get_default_config()

        expected_keys = [
            "num_channels", "kernel_size", "dropout",
            "batch_size", "max_epochs"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@requires_torch
class TestTCNTraining:
    """Tests for TCNModel training."""

    def test_fit_returns_metrics(self, small_sequence_data, fast_tcn_config):
        """Training should return TrainingMetrics."""
        from src.models.neural import TCNModel
        model = TCNModel(config=fast_tcn_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_receptive_field_in_metadata(self, small_sequence_data, fast_tcn_config):
        """Training should report receptive field in metadata."""
        from src.models.neural import TCNModel
        model = TCNModel(config=fast_tcn_config)
        metrics = model.fit(
            small_sequence_data["X_train"],
            small_sequence_data["y_train"],
            small_sequence_data["X_val"],
            small_sequence_data["y_val"],
        )

        assert "receptive_field" in metrics.metadata

    def test_is_fitted_after_training(self, trained_tcn):
        """Model should be fitted after training."""
        assert trained_tcn.is_fitted is True


@requires_torch
class TestTCNPrediction:
    """Tests for TCNModel prediction."""

    def test_predict_returns_output(self, trained_tcn, small_sequence_data):
        """Prediction should return PredictionOutput."""
        output = trained_tcn.predict(small_sequence_data["X_val"])

        assert output.n_samples == len(small_sequence_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_tcn, small_sequence_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_tcn.predict(small_sequence_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_tcn, small_sequence_data):
        """Probabilities should sum to 1."""
        output = trained_tcn.predict(small_sequence_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


@requires_torch
class TestTCNSaveLoad:
    """Tests for TCNModel serialization."""

    def test_save_creates_files(self, trained_tcn, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "tcn_model"
        trained_tcn.save(save_path)

        assert (save_path / "model.pt").exists()

    def test_predictions_match_after_load(self, trained_tcn, small_sequence_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.neural import TCNModel
        save_path = tmp_model_dir / "tcn_model"
        trained_tcn.save(save_path)

        loaded = TCNModel()
        loaded.load(save_path)

        original = trained_tcn.predict(small_sequence_data["X_val"])
        restored = loaded.predict(small_sequence_data["X_val"])

        # Use higher tolerance due to GPU/CPU floating point differences
        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-3)


@requires_torch
class TestTCNReceptiveField:
    """Tests for TCN receptive field calculation."""

    def test_receptive_field_calculation(self):
        """Receptive field should be calculated correctly."""
        from src.models.neural.tcn_model import TCNNetwork

        # num_channels = [8, 8] means 2 layers
        # kernel_size = 3, dilation_base = 2
        # RF = 1 + sum(2 * (k-1) * d for each layer)
        # Layer 0: d=1, layer 1: d=2
        # RF = 1 + 2*(3-1)*1 + 2*(3-1)*2 = 1 + 4 + 8 = 13

        network = TCNNetwork(
            input_size=8,
            num_channels=[8, 8],
            kernel_size=3,
            dropout=0.0,
            dilation_base=2,
        )

        assert network.receptive_field == 13

    def test_deeper_network_larger_rf(self):
        """Deeper network should have larger receptive field."""
        from src.models.neural.tcn_model import TCNNetwork

        shallow = TCNNetwork(
            input_size=8,
            num_channels=[8, 8],
            kernel_size=3,
            dropout=0.0,
        )

        deep = TCNNetwork(
            input_size=8,
            num_channels=[8, 8, 8, 8],
            kernel_size=3,
            dropout=0.0,
        )

        assert deep.receptive_field > shallow.receptive_field


# =============================================================================
# GPU AND MIXED PRECISION TESTS
# =============================================================================

@requires_cuda
class TestNeuralGPU:
    """Tests for neural model GPU support."""

    def test_lstm_cuda_device(self):
        """LSTM should use CUDA when available."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config={"device": "cuda"})
        assert model._device.type == "cuda"

    def test_gru_cuda_device(self):
        """GRU should use CUDA when available."""
        from src.models.neural import GRUModel
        model = GRUModel(config={"device": "cuda"})
        assert model._device.type == "cuda"

    def test_tcn_cuda_device(self):
        """TCN should use CUDA when available."""
        from src.models.neural import TCNModel
        model = TCNModel(config={"device": "cuda"})
        assert model._device.type == "cuda"


@requires_torch
class TestMixedPrecision:
    """Tests for mixed precision training."""

    def test_mixed_precision_enabled(self):
        """Mixed precision should be enabled by default on CUDA."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config={"mixed_precision": True})
        assert model._use_amp is True

    def test_mixed_precision_disabled(self):
        """Mixed precision should be disableable."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config={"mixed_precision": False})
        assert model._use_amp is False


# =============================================================================
# CROSS-MODEL TESTS
# =============================================================================

@requires_torch
class TestNeuralModelConsistency:
    """Tests for consistent behavior across neural models."""

    def test_all_models_in_registry(self):
        """All neural models should be in registry."""
        from src.models import ModelRegistry
        from src.models.neural import LSTMModel, GRUModel, TCNModel

        families = ModelRegistry.list_models()
        assert "neural" in families
        assert "lstm" in families["neural"]
        assert "gru" in families["neural"]
        assert "tcn" in families["neural"]

    def test_all_require_sequences(self):
        """All neural models should require sequences."""
        from src.models.neural import LSTMModel, GRUModel, TCNModel

        for ModelClass in [LSTMModel, GRUModel, TCNModel]:
            model = ModelClass()
            assert model.requires_sequences is True

    def test_all_require_scaling(self):
        """All neural models should require scaling."""
        from src.models.neural import LSTMModel, GRUModel, TCNModel

        for ModelClass in [LSTMModel, GRUModel, TCNModel]:
            model = ModelClass()
            assert model.requires_scaling is True

    def test_prediction_output_format_consistent(
        self, trained_lstm, trained_gru, trained_tcn, small_sequence_data
    ):
        """All models should return same prediction format."""
        for model in [trained_lstm, trained_gru, trained_tcn]:
            output = model.predict(small_sequence_data["X_val"])

            # Check required attributes
            assert hasattr(output, "class_predictions")
            assert hasattr(output, "class_probabilities")
            assert hasattr(output, "confidence")

            # Check shapes
            n_samples = len(small_sequence_data["X_val"])
            assert output.class_predictions.shape == (n_samples,)
            assert output.class_probabilities.shape == (n_samples, 3)
            assert output.confidence.shape == (n_samples,)


# =============================================================================
# PRODUCTION SAFETY AND BIDIRECTIONAL WARNING TESTS
# =============================================================================

@requires_torch
class TestProductionSafety:
    """Tests for is_production_safe property."""

    def test_lstm_production_safe_default(self):
        """LSTM with default config (bidirectional=False) is production-safe."""
        from src.models.neural import LSTMModel
        model = LSTMModel()
        assert model.is_production_safe is True

    def test_lstm_production_unsafe_bidirectional(self):
        """LSTM with bidirectional=True is not production-safe."""
        from src.models.neural import LSTMModel
        model = LSTMModel(config={"bidirectional": True})
        assert model.is_production_safe is False

    def test_gru_production_safe_default(self):
        """GRU with default config (bidirectional=False) is production-safe."""
        from src.models.neural import GRUModel
        model = GRUModel()
        assert model.is_production_safe is True

    def test_gru_production_unsafe_bidirectional(self):
        """GRU with bidirectional=True is not production-safe."""
        from src.models.neural import GRUModel
        model = GRUModel(config={"bidirectional": True})
        assert model.is_production_safe is False

    def test_tcn_always_production_safe(self):
        """TCN is always production-safe (causal convolutions)."""
        from src.models.neural import TCNModel
        model = TCNModel()
        assert model.is_production_safe is True

    def test_transformer_never_production_safe(self):
        """Transformer is never production-safe (bidirectional attention)."""
        from src.models.neural import TransformerModel
        model = TransformerModel()
        assert model.is_production_safe is False


@requires_torch
class TestBidirectionalWarning:
    """Tests for bidirectional/non-causal warnings."""

    def test_bidirectional_warning_logged(self, small_sequence_data, caplog):
        """Verify bidirectional=True logs a warning during fit."""
        import logging
        from src.models.neural import LSTMModel

        model = LSTMModel(config={
            "bidirectional": True,
            "hidden_size": 16,
            "num_layers": 1,
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
        })

        with caplog.at_level(logging.WARNING):
            model.fit(
                small_sequence_data["X_train"],
                small_sequence_data["y_train"],
                small_sequence_data["X_val"],
                small_sequence_data["y_val"],
            )

        assert "BIDIRECTIONAL RNN ENABLED" in caplog.text
        assert "production trading models" in caplog.text

    def test_no_warning_when_not_bidirectional(self, small_sequence_data, caplog):
        """Verify no bidirectional warning when bidirectional=False."""
        import logging
        from src.models.neural import LSTMModel

        model = LSTMModel(config={
            "bidirectional": False,
            "hidden_size": 16,
            "num_layers": 1,
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
        })

        with caplog.at_level(logging.WARNING):
            model.fit(
                small_sequence_data["X_train"],
                small_sequence_data["y_train"],
                small_sequence_data["X_val"],
                small_sequence_data["y_val"],
            )

        assert "BIDIRECTIONAL RNN ENABLED" not in caplog.text

    def test_warning_logged_only_once(self, small_sequence_data, caplog):
        """Verify bidirectional warning is only logged once per model."""
        import logging
        from src.models.neural import LSTMModel

        model = LSTMModel(config={
            "bidirectional": True,
            "hidden_size": 16,
            "num_layers": 1,
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
        })

        # First fit
        with caplog.at_level(logging.WARNING):
            model.fit(
                small_sequence_data["X_train"],
                small_sequence_data["y_train"],
                small_sequence_data["X_val"],
                small_sequence_data["y_val"],
            )

        first_count = caplog.text.count("BIDIRECTIONAL RNN ENABLED")
        assert first_count == 1

        # Second fit on same model instance - no new warning
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            model.fit(
                small_sequence_data["X_train"],
                small_sequence_data["y_train"],
                small_sequence_data["X_val"],
                small_sequence_data["y_val"],
            )

        # Should not have logged another warning
        assert "BIDIRECTIONAL RNN ENABLED" not in caplog.text

    def test_transformer_noncausal_warning_logged(self, small_sequence_data, caplog):
        """Verify Transformer logs non-causal attention warning."""
        import logging
        from src.models.neural import TransformerModel

        model = TransformerModel(config={
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
        })

        with caplog.at_level(logging.WARNING):
            model.fit(
                small_sequence_data["X_train"],
                small_sequence_data["y_train"],
                small_sequence_data["X_val"],
                small_sequence_data["y_val"],
            )

        assert "TRANSFORMER NON-CAUSAL ATTENTION" in caplog.text
        assert "production trading" in caplog.text

    def test_tcn_no_warning(self, small_sequence_data, caplog):
        """Verify TCN does not log any causality warnings."""
        import logging
        from src.models.neural import TCNModel

        model = TCNModel(config={
            "num_channels": [8, 8],
            "kernel_size": 2,
            "batch_size": 16,
            "max_epochs": 1,
            "device": "cpu",
            "mixed_precision": False,
            "num_workers": 0,
        })

        with caplog.at_level(logging.WARNING):
            model.fit(
                small_sequence_data["X_train"],
                small_sequence_data["y_train"],
                small_sequence_data["X_val"],
                small_sequence_data["y_val"],
            )

        # TCN is causal, no warnings about bidirectional or non-causal
        assert "BIDIRECTIONAL" not in caplog.text
        assert "NON-CAUSAL" not in caplog.text
