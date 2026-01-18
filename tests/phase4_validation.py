#!/usr/bin/env python3
"""
PHASE_4 Ensemble/Meta-Learners Comprehensive Validation Test.

Tests ALL PHASE_4 components:
1. Meta-Learners: ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
2. EnsembleOrchestrator and EnsembleResult
3. Stacking dataset building
4. OOF alignment for heterogeneous models
5. Model Registry integration
6. Integration with PHASE_2 OOFAligner and PHASE_3 CV results
7. PipelineConfig integration
"""

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class Phase4Validator:
    """Comprehensive PHASE_4 validation test suite."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.import_errors: List[str] = []
        self.available_meta_learners: List[str] = []
        self.components_tested: int = 0

    def run_all_tests(self) -> None:
        """Run all PHASE_4 validation tests."""
        print("=" * 70)
        print("PHASE_4 (Ensemble/Meta-Learners) COMPREHENSIVE VALIDATION")
        print("=" * 70)
        print()

        # 1. Test imports
        self._test_imports()

        # 2. Test Meta-Learners
        self._test_meta_learners()

        # 3. Test Meta-Learner Factory
        self._test_meta_learner_factory()

        # 4. Test EnsembleOrchestrator
        self._test_ensemble_orchestrator()

        # 5. Test EnsembleResult dataclass
        self._test_ensemble_result()

        # 6. Test Stacking module
        self._test_stacking_ensemble()

        # 7. Test Heterogeneous Stacking Builder
        self._test_heterogeneous_stacking_builder()

        # 8. Test Model Registry integration
        self._test_model_registry()

        # 9. Test OOF Alignment (PHASE_2 integration)
        self._test_oof_alignment()

        # 10. Test PipelineConfig integration
        self._test_pipeline_config()

        # 11. Test Diversity Analysis
        self._test_diversity_analysis()

        # Generate summary report
        self._generate_report()

    def _test_imports(self) -> None:
        """Test all required imports."""
        print("Testing imports...")

        imports_to_test = [
            # Meta-learners
            ("src.models.ensemble.ridge_meta", "RidgeMetaLearner"),
            ("src.models.ensemble.mlp_meta", "MLPMetaLearner"),
            ("src.models.ensemble.xgboost_meta", "XGBoostMeta"),
            ("src.models.ensemble.calibrated_meta", "CalibratedMetaLearner"),
            # Meta-learner factory
            ("src.models.ensemble.meta_factory", "MetaLearnerFactory"),
            ("src.models.ensemble.meta_factory", "get_meta_learner"),
            ("src.models.ensemble.meta_factory", "list_meta_learners"),
            # Orchestrator
            ("src.models.ensemble.orchestrator", "EnsembleOrchestrator"),
            ("src.models.ensemble.orchestrator", "EnsembleResult"),
            ("src.models.ensemble.orchestrator", "build_ensemble"),
            # Stacking
            ("src.models.ensemble.stacking", "StackingEnsemble"),
            ("src.models.ensemble.heterogeneous_stacking", "HeterogeneousStackingBuilder"),
            ("src.models.ensemble.heterogeneous_stacking", "StackingFeatures"),
            ("src.models.ensemble.heterogeneous_stacking", "build_stacking_features"),
            # Diversity
            ("src.models.ensemble.diversity", "DiversityAnalyzer"),
            ("src.models.ensemble.diversity", "DiversityMetrics"),
            # Ensemble package
            ("src.models.ensemble", "VotingEnsemble"),
            ("src.models.ensemble", "BlendingEnsemble"),
            # Registry
            ("src.models.registry", "ModelRegistry"),
            # Core interfaces
            ("src.core", "PipelineConfig"),
            ("src.core", "OOFResult"),
            ("src.core", "MODEL_DATA_RANKS"),
            ("src.core", "MODEL_ADAPTER_MAP"),
            # Adapters
            ("src.adapters", "OOFAligner"),
            ("src.adapters", "AlignedOOFResult"),
            ("src.adapters", "align_oof_predictions"),
            # Cross-validation
            ("src.cross_validation", "OOFPrediction"),
            ("src.cross_validation", "StackingDataset"),
        ]

        for module_name, class_name in imports_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                obj = getattr(module, class_name)
                self.results.append(TestResult(
                    name=f"Import {module_name}.{class_name}",
                    passed=True,
                    details={"type": type(obj).__name__}
                ))
                self.components_tested += 1
            except ImportError as e:
                self.import_errors.append(f"{module_name}.{class_name}: {e}")
                self.results.append(TestResult(
                    name=f"Import {module_name}.{class_name}",
                    passed=False,
                    error=str(e)
                ))
            except AttributeError as e:
                self.import_errors.append(f"{module_name}.{class_name}: {e}")
                self.results.append(TestResult(
                    name=f"Import {module_name}.{class_name}",
                    passed=False,
                    error=str(e)
                ))

        print(f"  Tested {len(imports_to_test)} imports")

    def _test_meta_learners(self) -> None:
        """Test all 4 meta-learners."""
        print("\nTesting Meta-Learners...")

        meta_learners = [
            ("ridge_meta", "RidgeMetaLearner"),
            ("mlp_meta", "MLPMetaLearner"),
            ("xgboost_meta", "XGBoostMeta"),
            ("calibrated_meta", "CalibratedMetaLearner"),
        ]

        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 500
        n_features = 9  # 3 models * 3 classes
        X_train = np.random.rand(n_samples, n_features)
        X_val = np.random.rand(n_samples // 5, n_features)
        y_train = np.random.choice([-1, 0, 1], size=n_samples)
        y_val = np.random.choice([-1, 0, 1], size=n_samples // 5)

        for meta_name, class_name in meta_learners:
            test_name = f"Meta-Learner {meta_name}"
            try:
                # Import and instantiate
                if meta_name == "ridge_meta":
                    from src.models.ensemble.ridge_meta import RidgeMetaLearner
                    meta = RidgeMetaLearner()
                elif meta_name == "mlp_meta":
                    from src.models.ensemble.mlp_meta import MLPMetaLearner
                    meta = MLPMetaLearner()
                elif meta_name == "xgboost_meta":
                    from src.models.ensemble.xgboost_meta import XGBoostMeta
                    meta = XGBoostMeta()
                elif meta_name == "calibrated_meta":
                    from src.models.ensemble.calibrated_meta import CalibratedMetaLearner
                    meta = CalibratedMetaLearner()

                # Test fit
                metrics = meta.fit(X_train, y_train, X_val, y_val)

                # Test predict
                output = meta.predict(X_val)

                # Validate output
                assert hasattr(output, "class_predictions"), "Missing class_predictions"
                assert hasattr(output, "class_probabilities"), "Missing class_probabilities"
                assert hasattr(output, "confidence"), "Missing confidence"
                assert len(output.class_predictions) == len(X_val)
                assert output.class_probabilities.shape == (len(X_val), 3)

                # Validate metrics
                assert hasattr(metrics, "train_loss"), "Missing train_loss"
                assert hasattr(metrics, "val_loss"), "Missing val_loss"
                assert hasattr(metrics, "val_f1"), "Missing val_f1"

                self.available_meta_learners.append(meta_name)
                self.results.append(TestResult(
                    name=test_name,
                    passed=True,
                    details={
                        "val_f1": float(metrics.val_f1),
                        "val_accuracy": float(metrics.val_accuracy),
                        "has_feature_importance": meta.get_feature_importance() is not None,
                    }
                ))
                self.components_tested += 1

            except Exception as e:
                self.results.append(TestResult(
                    name=test_name,
                    passed=False,
                    error=str(e)
                ))

        print(f"  Available meta-learners: {self.available_meta_learners}")

    def _test_meta_learner_factory(self) -> None:
        """Test MetaLearnerFactory functionality."""
        print("\nTesting Meta-Learner Factory...")

        test_name = "MetaLearnerFactory"
        try:
            from src.models.ensemble.meta_factory import (
                MetaLearnerFactory,
                get_meta_learner,
                list_meta_learners,
                create_meta_learner_from_config,
                META_LEARNER_REGISTRY,
            )

            # Test list_available
            available = MetaLearnerFactory.list_available()
            assert len(available) >= 4, f"Expected at least 4 meta-learners, got {len(available)}"

            # Test factory creation without config
            factory = MetaLearnerFactory()
            ridge = factory.create("ridge_meta")
            assert ridge is not None

            # Test factory with custom params
            mlp = factory.create("mlp_meta", hidden_layer_sizes=(32, 16))
            assert mlp is not None

            # Test get_meta_learner convenience function
            xgb = get_meta_learner("xgboost_meta", n_estimators=50)
            assert xgb is not None

            # Test get_meta_learner_info
            info = MetaLearnerFactory.get_meta_learner_info("ridge_meta")
            assert "description" in info
            assert "default_params" in info

            # Test get_all_info
            all_info = MetaLearnerFactory.get_all_info()
            assert len(all_info) >= 4

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "available_count": len(available),
                    "available": available,
                    "registry_size": len(META_LEARNER_REGISTRY),
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_ensemble_orchestrator(self) -> None:
        """Test EnsembleOrchestrator."""
        print("\nTesting EnsembleOrchestrator...")

        test_name = "EnsembleOrchestrator class"
        try:
            from src.models.ensemble.orchestrator import EnsembleOrchestrator

            # Check class attributes
            assert hasattr(EnsembleOrchestrator, "AVAILABLE_META_LEARNERS")
            assert len(EnsembleOrchestrator.AVAILABLE_META_LEARNERS) == 4
            assert "ridge_meta" in EnsembleOrchestrator.AVAILABLE_META_LEARNERS
            assert "mlp_meta" in EnsembleOrchestrator.AVAILABLE_META_LEARNERS
            assert "xgboost_meta" in EnsembleOrchestrator.AVAILABLE_META_LEARNERS
            assert "calibrated_meta" in EnsembleOrchestrator.AVAILABLE_META_LEARNERS

            # Check methods exist
            assert hasattr(EnsembleOrchestrator, "__init__")
            assert hasattr(EnsembleOrchestrator, "train")
            assert hasattr(EnsembleOrchestrator, "train_from_training_result")
            assert hasattr(EnsembleOrchestrator, "predict")
            assert hasattr(EnsembleOrchestrator, "predict_proba")
            assert hasattr(EnsembleOrchestrator, "load")
            assert hasattr(EnsembleOrchestrator, "_convert_to_oof_results")
            assert hasattr(EnsembleOrchestrator, "_save_results")

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "meta_learners": EnsembleOrchestrator.AVAILABLE_META_LEARNERS,
                    "has_train": True,
                    "has_train_from_training_result": True,
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

        # Test build_ensemble function
        test_name = "build_ensemble function"
        try:
            from src.models.ensemble.orchestrator import build_ensemble
            assert callable(build_ensemble)

            self.results.append(TestResult(
                name=test_name,
                passed=True,
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_ensemble_result(self) -> None:
        """Test EnsembleResult dataclass."""
        print("\nTesting EnsembleResult dataclass...")

        test_name = "EnsembleResult dataclass"
        try:
            from src.models.ensemble.orchestrator import EnsembleResult

            # Create instance
            result = EnsembleResult(
                ensemble_name="test_ensemble",
                meta_learner_name="ridge_meta",
                base_model_names=["xgboost", "lightgbm"],
                metrics={"val_f1": 0.65, "val_accuracy": 0.70},
                training_time_seconds=10.5,
                n_base_models=2,
                coverage=0.98,
                alignment_offset=59,
            )

            # Test to_dict
            d = result.to_dict()
            assert "ensemble_name" in d
            assert "meta_learner_name" in d
            assert "base_model_names" in d
            assert "metrics" in d
            assert "coverage" in d

            # Test summary
            summary = result.summary()
            assert "EnsembleResult" in summary
            assert "ridge_meta" in summary

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "has_to_dict": True,
                    "has_summary": True,
                    "fields": list(d.keys()),
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_stacking_ensemble(self) -> None:
        """Test StackingEnsemble class."""
        print("\nTesting StackingEnsemble...")

        test_name = "StackingEnsemble class"
        try:
            from src.models.ensemble.stacking import StackingEnsemble

            # Check class exists and has required methods
            assert hasattr(StackingEnsemble, "fit")
            assert hasattr(StackingEnsemble, "predict")
            assert hasattr(StackingEnsemble, "predict_proba")
            assert hasattr(StackingEnsemble, "save")
            assert hasattr(StackingEnsemble, "load")
            assert hasattr(StackingEnsemble, "clear_cache")
            assert hasattr(StackingEnsemble, "get_cache_size")
            assert hasattr(StackingEnsemble, "get_diversity_metrics")
            assert hasattr(StackingEnsemble, "suggest_model_removal")

            # Test instantiation
            stacking = StackingEnsemble(config={
                "base_model_names": ["xgboost", "lightgbm"],
                "meta_learner_name": "logistic",
                "n_folds": 3,
            })
            assert stacking.ensemble_type == "stacking"

            # Check heterogeneous support (instance attribute)
            assert hasattr(stacking, "_is_heterogeneous")

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "has_diversity_analysis": True,
                    "has_heterogeneous_support": True,
                    "has_cache_management": True,
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_heterogeneous_stacking_builder(self) -> None:
        """Test HeterogeneousStackingBuilder."""
        print("\nTesting HeterogeneousStackingBuilder...")

        test_name = "HeterogeneousStackingBuilder"
        try:
            from src.models.ensemble.heterogeneous_stacking import (
                HeterogeneousStackingBuilder,
                StackingFeatures,
                build_stacking_features,
            )

            # Test class instantiation
            builder = HeterogeneousStackingBuilder(add_derived_features=True)
            assert builder.add_derived_features is True
            assert builder.class_names == ["short", "neutral", "long"]

            # Test StackingFeatures dataclass
            sf = StackingFeatures(
                X=np.zeros((100, 12)),
                y=np.zeros(100),
                feature_names=["f1", "f2"] * 6,
                model_names=["xgboost", "lstm"],
                n_samples=100,
                n_features=12,
                coverage=0.95,
                alignment_offset=59,
            )
            assert sf.n_base_models == 2
            assert sf.n_prob_features == 6  # 2 models * 3 classes

            # Test derived feature count
            assert sf.n_derived_features == 3  # mean_confidence, agreement, entropy

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "has_from_config": hasattr(builder, "from_config"),
                    "has_build": hasattr(builder, "build"),
                    "has_coverage_report": hasattr(builder, "get_model_coverage_report"),
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_model_registry(self) -> None:
        """Test Model Registry integration."""
        print("\nTesting Model Registry integration...")

        test_name = "Model Registry meta-learner registration"
        try:
            from src.models.registry import ModelRegistry

            # Check meta-learners are registered
            registered = []
            for name in ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"]:
                if ModelRegistry.is_registered(name):
                    registered.append(name)

            # Get ensemble family models
            try:
                ensemble_models = ModelRegistry.list_family("ensemble")
            except ValueError:
                ensemble_models = []

            # Test create meta-learner via registry
            if ModelRegistry.is_registered("ridge_meta"):
                meta = ModelRegistry.create("ridge_meta")
                assert meta is not None

            self.results.append(TestResult(
                name=test_name,
                passed=len(registered) >= 4,
                details={
                    "registered_meta_learners": registered,
                    "ensemble_family_models": ensemble_models,
                    "total_registered": ModelRegistry.count(),
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_oof_alignment(self) -> None:
        """Test PHASE_2 OOFAligner integration."""
        print("\nTesting OOF Alignment (PHASE_2 integration)...")

        test_name = "OOFAligner"
        try:
            from src.adapters import OOFAligner, AlignedOOFResult
            from src.core import OOFResult

            # Create synthetic OOF results
            n_total = 1000
            n_sequence = 941  # n_total - seq_len + 1

            # Tabular model OOF (full coverage)
            oof_tabular = OOFResult(
                predictions=np.random.choice([-1, 0, 1], size=n_total),
                probabilities=np.random.dirichlet([1, 1, 1], size=n_total),
                indices=np.arange(n_total),
                fold_ids=np.zeros(n_total, dtype=int),
                model_name="xgboost",
                coverage=1.0,
            )

            # Sequence model OOF (partial coverage)
            oof_sequence = OOFResult(
                predictions=np.random.choice([-1, 0, 1], size=n_sequence),
                probabilities=np.random.dirichlet([1, 1, 1], size=n_sequence),
                indices=np.arange(59, n_total),  # Offset by seq_len - 1
                fold_ids=np.zeros(n_sequence, dtype=int),
                model_name="lstm",
                coverage=n_sequence / n_total,
            )

            # Test alignment
            aligner = OOFAligner()
            aligned = aligner.align([oof_tabular, oof_sequence], strategy="intersection")

            # Validate alignment result
            assert isinstance(aligned, AlignedOOFResult)
            assert aligned.n_common <= n_sequence
            assert len(aligned.model_names) == 2
            assert "xgboost" in aligned.model_names
            assert "lstm" in aligned.model_names

            # Test stacking features
            stacking_X = aligned.stacking_features
            assert stacking_X.shape[0] == aligned.n_common

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "n_common": aligned.n_common,
                    "coverage": dict(aligned.coverage),
                    "stacking_features_shape": stacking_X.shape,
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_pipeline_config(self) -> None:
        """Test PipelineConfig integration."""
        print("\nTesting PipelineConfig integration...")

        test_name = "PipelineConfig meta_learner validation"
        try:
            from src.core import PipelineConfig

            # Test valid meta-learner configs
            for meta in ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"]:
                config = PipelineConfig(
                    symbol="TEST",
                    data_path="/tmp/test.parquet",
                    output_dir="/tmp/test_output",
                    models=["xgboost", "lightgbm"],
                    meta_learner=meta,
                )
                assert config.meta_learner == meta

            # Test invalid meta-learner raises error
            try:
                bad_config = PipelineConfig(
                    symbol="TEST",
                    data_path="/tmp/test.parquet",
                    output_dir="/tmp/test_output",
                    meta_learner="invalid_meta",
                )
                validation_works = False
            except Exception:
                validation_works = True

            self.results.append(TestResult(
                name=test_name,
                passed=validation_works,
                details={
                    "valid_meta_learners": ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"],
                    "validation_works": validation_works,
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _test_diversity_analysis(self) -> None:
        """Test diversity analysis components."""
        print("\nTesting Diversity Analysis...")

        test_name = "DiversityAnalyzer"
        try:
            from src.models.ensemble.diversity import DiversityAnalyzer, DiversityMetrics

            # Create synthetic predictions
            np.random.seed(42)
            n_samples = 500
            n_models = 3

            base_predictions = {
                "xgboost": np.random.choice([-1, 0, 1], size=n_samples),
                "lightgbm": np.random.choice([-1, 0, 1], size=n_samples),
                "catboost": np.random.choice([-1, 0, 1], size=n_samples),
            }
            base_probabilities = {
                name: np.random.dirichlet([1, 1, 1], size=n_samples)
                for name in base_predictions
            }
            y_true = np.random.choice([-1, 0, 1], size=n_samples)

            # Test analyzer
            analyzer = DiversityAnalyzer(
                min_diversity_threshold=0.3,
                correlation_threshold=0.8,
            )
            metrics = analyzer.analyze(
                base_predictions=base_predictions,
                base_probabilities=base_probabilities,
                y_true=y_true,
            )

            # Validate metrics
            assert isinstance(metrics, DiversityMetrics)
            assert hasattr(metrics, "diversity_score")
            assert hasattr(metrics, "pairwise_correlation")
            assert hasattr(metrics, "disagreement")
            assert hasattr(metrics, "q_statistic")
            assert hasattr(metrics, "recommendations")

            # Test to_dict
            d = metrics.to_dict()
            assert "diversity_score" in d

            self.results.append(TestResult(
                name=test_name,
                passed=True,
                details={
                    "diversity_score": float(metrics.diversity_score),
                    "pairwise_correlation": float(metrics.pairwise_correlation),
                    "disagreement": float(metrics.disagreement),
                    "n_recommendations": len(metrics.recommendations),
                }
            ))
            self.components_tested += 1

        except Exception as e:
            self.results.append(TestResult(
                name=test_name,
                passed=False,
                error=str(e)
            ))

    def _generate_report(self) -> None:
        """Generate final validation report."""
        print("\n" + "=" * 70)
        print("PHASE_4 VALIDATION REPORT")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        print(f"\nTotal components tested: {self.components_tested}")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
        print(f"Available meta-learners: {len(self.available_meta_learners)}")

        if self.import_errors:
            print(f"\nImport errors: {len(self.import_errors)}")
            for err in self.import_errors[:5]:  # Show first 5
                print(f"  - {err}")

        print("\n" + "-" * 70)
        print("DETAILED RESULTS")
        print("-" * 70)

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status} {result.name}")
            if result.error:
                print(f"       Error: {result.error}")
            if result.details and result.passed:
                for key, value in result.details.items():
                    if isinstance(value, float):
                        print(f"       {key}: {value:.4f}")
                    else:
                        print(f"       {key}: {value}")

        print("\n" + "-" * 70)
        print("PHASE_4 HEALTH STATUS")
        print("-" * 70)

        # Determine overall health
        if failed == 0 and len(self.available_meta_learners) == 4:
            health = "HEALTHY - All components operational"
            status_symbol = "[OK]"
        elif failed <= 2 and len(self.available_meta_learners) >= 3:
            health = "PARTIAL - Minor issues detected"
            status_symbol = "[WARN]"
        else:
            health = "DEGRADED - Significant issues found"
            status_symbol = "[ERROR]"

        print(f"\n{status_symbol} {health}")
        print(f"\nAvailable Meta-Learners: {self.available_meta_learners}")

        # Summary of key PHASE_4 components
        key_components = [
            "Meta-Learner ridge_meta",
            "Meta-Learner mlp_meta",
            "Meta-Learner xgboost_meta",
            "Meta-Learner calibrated_meta",
            "EnsembleOrchestrator class",
            "EnsembleResult dataclass",
            "StackingEnsemble class",
            "HeterogeneousStackingBuilder",
            "OOFAligner",
            "DiversityAnalyzer",
        ]

        print("\nKey Component Status:")
        for comp in key_components:
            found = any(r.name == comp or comp in r.name for r in self.results)
            passed = any(r.name == comp or comp in r.name and r.passed for r in self.results)
            if passed:
                print(f"  [OK] {comp}")
            elif found:
                print(f"  [FAIL] {comp}")
            else:
                print(f"  [--] {comp} (not tested)")

        print("\n" + "=" * 70)


def main():
    """Run PHASE_4 validation."""
    validator = Phase4Validator()
    validator.run_all_tests()


if __name__ == "__main__":
    main()
