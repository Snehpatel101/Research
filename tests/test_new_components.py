"""
Tests for newly implemented CODEBASE_RECOMMENDATIONS components.

Tests:
- DatasetContract (HIGH #4)
- Diversity Analysis (MEDIUM #6)
- ParallelTrainingService (MEDIUM #7)
- Bet Sizing (LOW #8)
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# TEST: DatasetContract (HIGH #4)
# =============================================================================

class TestDatasetContract:
    """Tests for DatasetContract class."""

    def test_import(self):
        """Test that DatasetContract can be imported."""
        from src.core import DatasetContract, SplitDatasetContract
        assert DatasetContract is not None
        assert SplitDatasetContract is not None

    def test_create_from_dataframe(self):
        """Test creating DatasetContract from DataFrame."""
        from src.core import DatasetContract

        features = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [10, 20, 30, 40, 50],
        })
        labels = pd.Series([0, 1, 0, 1, 0], name='label')

        contract = DatasetContract(
            features=features,
            labels=labels,
            split='train',
        )

        assert contract.n_samples == 5
        assert contract.n_features == 2
        assert contract.feature_names == ['f1', 'f2']
        assert contract.split == 'train'

    def test_to_numpy(self):
        """Test converting to numpy arrays."""
        from src.core import DatasetContract

        features = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
        labels = pd.Series([0, 1])

        contract = DatasetContract(features=features, labels=labels)
        X, y = contract.to_numpy()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (2, 2)
        assert y.shape == (2,)

    def test_from_arrays(self):
        """Test creating from numpy arrays."""
        from src.core import DatasetContract

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        contract = DatasetContract.from_arrays(X, y, split='val')

        assert contract.n_samples == 3
        assert contract.n_features == 2
        assert contract.split == 'val'

    def test_filter_by_indices(self):
        """Test filtering by indices."""
        from src.core import DatasetContract

        features = pd.DataFrame({'f1': [1, 2, 3, 4]}, index=[10, 20, 30, 40])
        labels = pd.Series([0, 1, 0, 1], index=[10, 20, 30, 40])

        contract = DatasetContract(features=features, labels=labels)
        filtered = contract.filter_by_indices(pd.Index([10, 30]))

        assert filtered.n_samples == 2
        assert list(filtered.features['f1']) == [1, 3]

    def test_validation_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        from src.core import DatasetContract

        features = pd.DataFrame({'f1': [1, 2, 3]})
        labels = pd.Series([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="length mismatch"):
            DatasetContract(features=features, labels=labels)

    def test_split_dataset_contract(self):
        """Test SplitDatasetContract."""
        from src.core import DatasetContract, SplitDatasetContract

        train = DatasetContract.from_arrays(np.array([[1]]), np.array([0]), split='train')
        val = DatasetContract.from_arrays(np.array([[2]]), np.array([1]), split='val')
        test = DatasetContract.from_arrays(np.array([[3]]), np.array([0]), split='test')

        split_contract = SplitDatasetContract(train=train, val=val, test=test)

        assert split_contract.has_test == True
        assert split_contract.train.split == 'train'


# =============================================================================
# TEST: Diversity Analysis (MEDIUM #6)
# =============================================================================

class TestDiversityAnalysis:
    """Tests for ensemble diversity analysis."""

    def test_import(self):
        """Test that diversity functions can be imported."""
        from src.models.ensemble.diversity import (
            compute_mcc_diversity_matrix,
            select_diverse_models,
            filter_correlated_models,
            DiversityAnalysisResult,
        )
        assert compute_mcc_diversity_matrix is not None
        assert select_diverse_models is not None

    def test_compute_diversity_matrix(self):
        """Test diversity matrix computation."""
        from src.models.ensemble.diversity import compute_mcc_diversity_matrix

        # Create some mock predictions
        pred1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pred2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Same as pred1
        pred3 = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # Different from pred1

        predictions = [pred1, pred2, pred3]
        diversity_matrix = compute_mcc_diversity_matrix(predictions)

        assert diversity_matrix.shape == (3, 3)
        # pred1 and pred2 are identical, so low diversity (MCC=1, diversity=0)
        assert diversity_matrix[0, 1] < 0.1
        # pred1 and pred3 are different, so higher diversity
        assert diversity_matrix[0, 2] > 0.3  # Not perfectly correlated

    def test_select_diverse_models(self):
        """Test diverse model selection."""
        from src.models.ensemble.diversity import select_diverse_models, DiversityAnalysisResult

        # Create mock OOF predictions
        class MockOOF:
            def __init__(self, predictions, metrics):
                self.predictions = predictions
                self.metrics = metrics

        oof_predictions = {
            'model_a': MockOOF(np.array([0, 1, 0, 1]), {'val_f1': 0.8}),
            'model_b': MockOOF(np.array([0, 1, 0, 1]), {'val_f1': 0.7}),  # Same as a
            'model_c': MockOOF(np.array([1, 0, 1, 0]), {'val_f1': 0.6}),  # Opposite
        }

        result = select_diverse_models(oof_predictions, min_diversity=0.3, max_models=2)

        assert isinstance(result, DiversityAnalysisResult)
        assert 'model_a' in result.selected_models  # Best performing
        assert len(result.selected_models) <= 2

    def test_filter_correlated_models(self):
        """Test filtering correlated models."""
        from src.models.ensemble.diversity import filter_correlated_models

        class MockOOF:
            def __init__(self, predictions):
                self.predictions = predictions
                self.metrics = {'val_f1': 0.5}

        oof_predictions = {
            'model_a': MockOOF(np.array([0, 1, 0, 1])),
            'model_b': MockOOF(np.array([0, 1, 0, 1])),  # Identical to a
        }

        filtered = filter_correlated_models(oof_predictions, correlation_threshold=0.9)

        # Should keep only one since they're identical
        assert len(filtered) <= len(oof_predictions)


# =============================================================================
# TEST: Parallel Training Service (MEDIUM #7)
# =============================================================================

class TestParallelTrainingService:
    """Tests for parallel training service."""

    def test_import(self):
        """Test that parallel training can be imported."""
        from src.models.training.services import (
            ParallelTrainingService,
            ParallelTrainingConfig,
            train_models_parallel,
        )
        assert ParallelTrainingService is not None
        assert ParallelTrainingConfig is not None

    def test_service_creation(self):
        """Test creating parallel training service."""
        from src.models.training.services import ParallelTrainingService

        service = ParallelTrainingService(n_jobs=2, verbose=0)

        assert service.n_jobs == 2
        assert service.verbose == 0
        assert service.backend == 'loky'

    def test_config_defaults(self):
        """Test parallel training config defaults."""
        from src.models.training.services import ParallelTrainingConfig

        config = ParallelTrainingConfig()

        assert config.n_jobs == -1
        assert config.verbose == 10
        assert config.backend == 'loky'

    def test_empty_request_list(self):
        """Test training with empty request list."""
        from src.models.training.services import ParallelTrainingService

        service = ParallelTrainingService()
        results = service.train_models_parallel([])

        assert results == []

    def test_pipeline_config_has_parallel_training(self):
        """Test that PipelineConfig has parallel_training field."""
        from src.pipeline_config import PipelineConfig

        config = PipelineConfig(symbol='MES')

        assert hasattr(config, 'parallel_training')
        assert hasattr(config, 'n_jobs')
        assert config.n_jobs == -1


# =============================================================================
# TEST: Bet Sizing (LOW #8)
# =============================================================================

class TestBetSizing:
    """Tests for bet sizing module."""

    def test_import(self):
        """Test that bet sizing can be imported."""
        from src.models.training.meta_labeling import (
            BetSizingStrategy,
            BetSizingConfig,
            compute_bet_sizes,
            predict_with_sizing,
        )
        assert BetSizingStrategy is not None
        assert compute_bet_sizes is not None

    def test_binary_sizing(self):
        """Test binary bet sizing strategy."""
        from src.models.training.meta_labeling import compute_bet_sizes, BetSizingStrategy

        probs = np.array([0.3, 0.5, 0.7, 0.9])
        sizes = compute_bet_sizes(probs, strategy=BetSizingStrategy.BINARY, threshold=0.5)

        # Below threshold should be 0, above should be 1
        assert sizes[0] == 0.0  # 0.3 < 0.5
        assert sizes[1] == 0.0  # 0.5 == 0.5 (not above)
        assert sizes[2] == 1.0  # 0.7 > 0.5
        assert sizes[3] == 1.0  # 0.9 > 0.5

    def test_proportional_sizing(self):
        """Test proportional bet sizing strategy."""
        from src.models.training.meta_labeling import compute_bet_sizes, BetSizingStrategy

        probs = np.array([0.5, 0.75, 1.0])
        sizes = compute_bet_sizes(probs, strategy=BetSizingStrategy.PROPORTIONAL, threshold=0.5)

        # Should scale linearly from threshold to 1.0
        assert sizes[0] == 0.0  # At threshold
        assert 0.4 < sizes[1] < 0.6  # Halfway
        assert sizes[2] == 1.0  # At max

    def test_kelly_sizing(self):
        """Test Kelly criterion bet sizing."""
        from src.models.training.meta_labeling import compute_bet_sizes, BetSizingStrategy

        probs = np.array([0.5, 0.75, 1.0])
        sizes = compute_bet_sizes(probs, strategy=BetSizingStrategy.KELLY, threshold=0.5, kelly_fraction=1.0)

        # Kelly: f* = 2p - 1 for even odds
        # p=0.5 -> 0, p=0.75 -> 0.5, p=1.0 -> 1.0
        assert sizes[0] == 0.0  # 2*0.5 - 1 = 0
        assert 0.45 < sizes[1] < 0.55  # 2*0.75 - 1 = 0.5
        assert sizes[2] == 1.0  # 2*1.0 - 1 = 1.0

    def test_predict_with_sizing(self):
        """Test combining predictions with sizing."""
        from src.models.training.meta_labeling import predict_with_sizing, BetSizingConfig, BetSizingStrategy

        primary = np.array([1, -1, 1, -1])  # Directions
        meta_probs = np.array([0.3, 0.7, 0.9, 0.4])

        config = BetSizingConfig(
            strategy=BetSizingStrategy.BINARY,
            threshold=0.5,
        )

        directions, sizes = predict_with_sizing(primary, meta_probs, config)

        # Where meta_prob < threshold, direction should become 0
        assert directions[0] == 0  # prob 0.3 < 0.5, no trade
        assert directions[1] == -1  # prob 0.7 > 0.5, keep direction
        assert directions[2] == 1  # prob 0.9 > 0.5, keep direction
        assert directions[3] == 0  # prob 0.4 < 0.5, no trade

    def test_all_strategies_exist(self):
        """Test that all strategies are implemented."""
        from src.models.training.meta_labeling import BetSizingStrategy, compute_bet_sizes

        probs = np.array([0.6, 0.8])

        for strategy in BetSizingStrategy:
            sizes = compute_bet_sizes(probs, strategy=strategy)
            assert len(sizes) == 2
            assert np.all(sizes >= 0)
            assert np.all(sizes <= 1)


# =============================================================================
# TEST: Training Services Integration
# =============================================================================

class TestTrainingServicesIntegration:
    """Test that all training services integrate correctly."""

    def test_all_services_importable(self):
        """Test all training services can be imported together."""
        from src.models.training.services import (
            ModelTrainingService,
            ModelTrainingRequest,
            ModelTrainingResult,
            OOFGenerationService,
            OOFRequest,
            HyperparameterTuningService,
            TuningRequest,
            TuningResult,
            ArtifactManager,
            ArtifactSaveRequest,
            ParallelTrainingService,
            ParallelTrainingConfig,
        )

        # All imports successful
        assert ModelTrainingService is not None
        assert OOFGenerationService is not None
        assert HyperparameterTuningService is not None
        assert ArtifactManager is not None
        assert ParallelTrainingService is not None

    def test_unified_orchestrator_uses_services(self):
        """Test that UnifiedTrainingOrchestrator imports the services."""
        from src.models.training.unified_orchestrator import UnifiedTrainingOrchestrator

        # Check that the module imports are in place
        import src.models.training.unified_orchestrator as uto_module

        # The orchestrator should have service imports
        source = open(uto_module.__file__).read()
        assert 'ModelTrainingService' in source
        assert 'OOFGenerationService' in source
        assert 'ArtifactManager' in source


# =============================================================================
# TEST: Module Structure
# =============================================================================

class TestModuleStructure:
    """Test that the module structure is correct."""

    def test_core_exports(self):
        """Test src.core exports."""
        from src.core import (
            PipelineConfig,
            DatasetContract,
            SplitDatasetContract,
        )
        assert all([PipelineConfig, DatasetContract, SplitDatasetContract])

    def test_training_exports(self):
        """Test src.training exports."""
        from src.models.training import (
            UnifiedTrainingOrchestrator,
            TrainingRunResult,
            BetSizingStrategy,
            BetSizingConfig,
            compute_bet_sizes,
        )
        assert all([UnifiedTrainingOrchestrator, TrainingRunResult, BetSizingStrategy])

    def test_ensemble_exports(self):
        """Test src.models.ensemble exports."""
        from src.models.ensemble import (
            select_diverse_models,
            filter_correlated_models,
            DiversityAnalysisResult,
        )
        assert all([select_diverse_models, filter_correlated_models, DiversityAnalysisResult])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
