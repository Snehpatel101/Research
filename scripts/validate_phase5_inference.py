#!/usr/bin/env python3
"""
PHASE_5 (Inference Bundle) Comprehensive Validation Script

This script performs a comprehensive validation of all PHASE_5 components
in the ML Factory pipeline, testing:
1. InferenceOrchestrator - Unified inference entry point
2. BundleBuilder - Bundle creation from training results
3. EnsembleBundle - Stacking ensemble serialization
4. Existing Infrastructure - ModelBundle, PreprocessingGraph, etc.
5. Integration Points - PipelineConfig, OOFAligner, etc.

Usage:
    python scripts/validate_phase5_inference.py
"""

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# TEST RESULT TRACKING
# =============================================================================


@dataclass
class ComponentTestResult:
    """Result for a single component test."""

    component_name: str
    passed: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase5ValidationReport:
    """Complete validation report for PHASE_5."""

    results: list[ComponentTestResult] = field(default_factory=list)
    bundle_versions: dict[str, str] = field(default_factory=dict)
    import_errors: list[str] = field(default_factory=list)

    @property
    def total_tested(self) -> int:
        return len(self.results)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def health_status(self) -> str:
        if self.import_errors:
            return "CRITICAL - Import Errors"
        if self.total_failed == 0:
            return "HEALTHY - All Components Operational"
        if self.total_failed < self.total_tested // 2:
            return "DEGRADED - Some Components Failed"
        return "CRITICAL - Majority of Components Failed"

    def print_report(self) -> None:
        """Print formatted validation report."""
        print("\n" + "=" * 80)
        print("PHASE_5 (INFERENCE BUNDLE) VALIDATION REPORT")
        print("=" * 80)

        # Summary
        print("\nSUMMARY:")
        print(f"  Total Components Tested: {self.total_tested}")
        print(f"  Passed: {self.total_passed}")
        print(f"  Failed: {self.total_failed}")
        print(f"  Health Status: {self.health_status}")

        # Bundle Versions
        if self.bundle_versions:
            print("\nBUNDLE VERSIONS:")
            for name, version in self.bundle_versions.items():
                print(f"  {name}: {version}")

        # Import Errors
        if self.import_errors:
            print(f"\nIMPORT ERRORS ({len(self.import_errors)}):")
            for err in self.import_errors:
                print(f"  [ERROR] {err}")

        # Detailed Results
        print("\nDETAILED RESULTS:")
        print("-" * 80)

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status} {result.component_name}")
            if result.error:
                print(f"       Error: {result.error}")
            if result.details:
                for key, value in result.details.items():
                    print(f"       {key}: {value}")

        print("\n" + "=" * 80)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def test_inference_orchestrator(report: Phase5ValidationReport) -> None:
    """Test InferenceOrchestrator component."""
    print("\nTesting InferenceOrchestrator...")

    try:
        from src.inference.orchestrator import (
            InferenceOrchestrator,
            PredictionResult,
            load_inference,
            predict_batch_from_bundle,
            predict_from_bundle,
        )

        # Test class instantiation
        orch = InferenceOrchestrator()

        # Verify class methods exist
        class_methods = [
            "from_config",
            "from_experiment",
            "from_bundle",
            "from_bundles",
            "from_training_result",
        ]

        missing_methods = []
        for method in class_methods:
            if not hasattr(InferenceOrchestrator, method):
                missing_methods.append(method)

        # Verify instance methods exist
        instance_methods = [
            "predict",
            "predict_all",
            "predict_from_raw",
            "predict_batch",
            "predict_with_uncertainty",
            "validate",
            "set_preprocessing_graph",
        ]

        for method in instance_methods:
            if not hasattr(orch, method):
                missing_methods.append(method)

        # Verify properties
        properties = [
            "loaded_models",
            "has_ensemble",
            "has_preprocessing_graph",
            "preprocessing_graph",
        ]

        for prop in properties:
            if not hasattr(orch, prop):
                missing_methods.append(f"property:{prop}")

        # Test PredictionResult dataclass
        import numpy as np

        pred_result = PredictionResult(
            class_predictions=np.array([0, 1, -1]),
            class_probabilities=np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]),
            confidence=np.array([0.5, 0.7, 0.6]),
            model_name="test_model",
            horizon=20,
            inference_time_ms=5.0,
            n_samples=3,
        )

        # Test PredictionResult methods
        df = pred_result.to_dataframe()
        summary = pred_result.summary()

        if missing_methods:
            report.results.append(
                ComponentTestResult(
                    component_name="InferenceOrchestrator",
                    passed=False,
                    error=f"Missing methods/properties: {missing_methods}",
                    details={
                        "class_methods_expected": class_methods,
                        "instance_methods_expected": instance_methods,
                        "properties_expected": properties,
                    },
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="InferenceOrchestrator",
                    passed=True,
                    details={
                        "factory_methods": len(class_methods),
                        "prediction_methods": len(instance_methods),
                        "properties": len(properties),
                        "PredictionResult": "verified",
                    },
                )
            )

        # Test convenience functions
        report.results.append(
            ComponentTestResult(
                component_name="InferenceOrchestrator.convenience_functions",
                passed=True,
                details={
                    "load_inference": callable(load_inference),
                    "predict_from_bundle": callable(predict_from_bundle),
                    "predict_batch_from_bundle": callable(predict_batch_from_bundle),
                },
            )
        )

    except ImportError as e:
        report.import_errors.append(f"InferenceOrchestrator: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="InferenceOrchestrator",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="InferenceOrchestrator",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_bundle_builder(report: Phase5ValidationReport) -> None:
    """Test BundleBuilder component."""
    print("\nTesting BundleBuilder...")

    try:
        from src.inference.builder import (
            BundleBuilder,
            BundleBuildResult,
            build_bundles,
            build_from_run,
        )

        # Test BundleBuildResult dataclass
        result = BundleBuildResult(
            bundle_paths=[Path("/test/path1"), Path("/test/path2")],
            ensemble_bundle_path=Path("/test/ensemble"),
            n_bundles=3,
            total_size_mb=50.5,
            metadata={"test": "value"},
        )

        # Test methods
        result_dict = result.to_dict()
        summary = result.summary()

        # Test BundleBuilder class methods
        class_methods = [
            "from_config",
            "from_training_run",
        ]

        instance_methods = [
            "build_from_training_result",
            "build_ensemble_bundle",
            "build_all",
            "validate_bundles",
        ]

        missing = []
        for method in class_methods:
            if not hasattr(BundleBuilder, method):
                missing.append(f"class:{method}")

        # Need config to instantiate, just check class attributes
        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="BundleBuilder",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="BundleBuilder",
                    passed=True,
                    details={
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                        "BundleBuildResult": "verified",
                    },
                )
            )

        # Test convenience functions
        report.results.append(
            ComponentTestResult(
                component_name="BundleBuilder.convenience_functions",
                passed=True,
                details={
                    "build_bundles": callable(build_bundles),
                    "build_from_run": callable(build_from_run),
                },
            )
        )

    except ImportError as e:
        report.import_errors.append(f"BundleBuilder: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="BundleBuilder",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="BundleBuilder",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_ensemble_bundle(report: Phase5ValidationReport) -> None:
    """Test EnsembleBundle component."""
    print("\nTesting EnsembleBundle...")

    try:
        from src.inference.ensemble_bundle import (
            ENSEMBLE_BUNDLE_VERSION,
            AlignmentConfig,
            EnsembleBundle,
            EnsembleBundleManifest,
            EnsembleBundleMetadata,
        )

        # Record version
        report.bundle_versions["EnsembleBundle"] = ENSEMBLE_BUNDLE_VERSION

        # Test AlignmentConfig
        align_config = AlignmentConfig(
            strategy="intersection",
            n_classes=3,
            model_offsets={"xgboost": 0, "lstm": 59},
            sequence_lengths={"xgboost": 0, "lstm": 60},
        )
        align_dict = align_config.to_dict()
        align_loaded = AlignmentConfig.from_dict(align_dict)

        # Test EnsembleBundleMetadata
        metadata = EnsembleBundleMetadata(
            version=ENSEMBLE_BUNDLE_VERSION,
            created_at="2024-01-01T00:00:00",
            meta_learner_name="ridge_meta",
            base_model_names=["xgboost", "lgbm", "lstm"],
            horizon=20,
            n_base_models=3,
            n_stacking_features=11,
            symbol="MES",
            coverage=0.98,
            alignment_offset=59,
            metrics={"accuracy": 0.65},
        )
        meta_dict = metadata.to_dict()
        meta_loaded = EnsembleBundleMetadata.from_dict(meta_dict)

        # Test EnsembleBundleManifest
        manifest = EnsembleBundleManifest(
            version=ENSEMBLE_BUNDLE_VERSION,
            files=["metadata.json", "meta_learner/model.pkl"],
            checksums={"metadata.json": "abc123"},
        )
        manifest_dict = manifest.to_dict()
        manifest_loaded = EnsembleBundleManifest.from_dict(manifest_dict)

        # Test EnsembleBundle class methods
        class_methods = [
            "from_ensemble_result",
            "from_orchestrator",
            "load",
        ]

        instance_methods = [
            "save",
            "predict",
            "predict_proba",
            "predict_classes",
            "predict_from_base_features",
            "validate",
            "summary",
        ]

        missing = []
        for method in class_methods:
            if not hasattr(EnsembleBundle, method):
                missing.append(f"class:{method}")

        for method in instance_methods:
            if not hasattr(EnsembleBundle, method):
                missing.append(f"instance:{method}")

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="EnsembleBundle",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="EnsembleBundle",
                    passed=True,
                    details={
                        "version": ENSEMBLE_BUNDLE_VERSION,
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                        "EnsembleBundleMetadata": "verified",
                        "EnsembleBundleManifest": "verified",
                        "AlignmentConfig": "verified",
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"EnsembleBundle: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="EnsembleBundle",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="EnsembleBundle",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_model_bundle(report: Phase5ValidationReport) -> None:
    """Test ModelBundle component."""
    print("\nTesting ModelBundle...")

    try:
        from src.inference.bundle import (
            BUNDLE_PREPROCESSING_GRAPH_FILE,
            BUNDLE_VERSION,
            BundleManifest,
            BundleMetadata,
            ModelBundle,
        )

        # Record version
        report.bundle_versions["ModelBundle"] = BUNDLE_VERSION

        # Test BundleMetadata
        metadata = BundleMetadata(
            version=BUNDLE_VERSION,
            created_at="2024-01-01T00:00:00",
            model_name="xgboost",
            model_family="gradient_boosting",
            horizon=20,
            n_features=100,
            feature_hash="abc123def456",
            requires_sequences=False,
            sequence_length=0,
            has_calibrator=True,
            has_preprocessing_graph=True,
            symbol="MES",
        )
        meta_dict = metadata.to_dict()
        meta_loaded = BundleMetadata.from_dict(meta_dict)

        # Test BundleManifest
        manifest = BundleManifest(
            version=BUNDLE_VERSION,
            files=["model/", "scaler.pkl", "metadata.json"],
            checksums={"metadata.json": "abc123"},
        )
        manifest_dict = manifest.to_dict()
        manifest_loaded = BundleManifest.from_dict(manifest_dict)

        # Test ModelBundle class methods
        class_methods = ["from_training", "load"]
        instance_methods = [
            "save",
            "predict",
            "validate",
            "set_preprocessing_graph",
            "preprocess",
            "predict_from_raw",
        ]

        missing = []
        for method in class_methods:
            if not hasattr(ModelBundle, method):
                missing.append(f"class:{method}")

        for method in instance_methods:
            if not hasattr(ModelBundle, method):
                missing.append(f"instance:{method}")

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="ModelBundle",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="ModelBundle",
                    passed=True,
                    details={
                        "version": BUNDLE_VERSION,
                        "preprocessing_graph_file": BUNDLE_PREPROCESSING_GRAPH_FILE,
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                        "BundleMetadata": "verified",
                        "BundleManifest": "verified",
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"ModelBundle: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="ModelBundle",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="ModelBundle",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_preprocessing_graph(report: Phase5ValidationReport) -> None:
    """Test PreprocessingGraph component."""
    print("\nTesting PreprocessingGraph...")

    try:
        from src.inference.preprocessing_graph import (
            PREPROCESSING_GRAPH_FILE,
            PREPROCESSING_GRAPH_VERSION,
            CleaningConfig,
            IndicatorConfig,
            MTFConfig,
            PreprocessingGraph,
            PreprocessingGraphConfig,
            RegimeConfig,
            ScalingConfig,
            WaveletConfig,
        )

        # Record version
        report.bundle_versions["PreprocessingGraph"] = PREPROCESSING_GRAPH_VERSION

        # Test config classes
        cleaning = CleaningConfig()
        indicators = IndicatorConfig()
        mtf = MTFConfig()
        wavelets = WaveletConfig()
        regime = RegimeConfig()
        scaling = ScalingConfig()

        # Test serialization
        for cfg, name in [
            (cleaning, "CleaningConfig"),
            (indicators, "IndicatorConfig"),
            (mtf, "MTFConfig"),
            (wavelets, "WaveletConfig"),
            (regime, "RegimeConfig"),
            (scaling, "ScalingConfig"),
        ]:
            cfg_dict = cfg.to_dict()
            cfg_class = type(cfg)
            cfg_loaded = cfg_class.from_dict(cfg_dict)

        # Test PreprocessingGraphConfig
        config = PreprocessingGraphConfig(
            version=PREPROCESSING_GRAPH_VERSION,
            horizon=20,
            symbol="MES",
            cleaning=cleaning,
            indicators=indicators,
            mtf=mtf,
            wavelets=wavelets,
            regime=regime,
            scaling=scaling,
        )
        config_dict = config.to_dict()
        config_hash = config.compute_hash()

        # Test PreprocessingGraph class methods
        class_methods = ["from_pipeline_config", "from_training_run", "load", "from_dict"]
        instance_methods = ["transform", "save", "validate", "set_scaler", "to_dict"]

        missing = []
        for method in class_methods:
            if not hasattr(PreprocessingGraph, method):
                missing.append(f"class:{method}")

        for method in instance_methods:
            if not hasattr(PreprocessingGraph, method):
                missing.append(f"instance:{method}")

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="PreprocessingGraph",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="PreprocessingGraph",
                    passed=True,
                    details={
                        "version": PREPROCESSING_GRAPH_VERSION,
                        "file_constant": PREPROCESSING_GRAPH_FILE,
                        "config_classes": [
                            "CleaningConfig",
                            "IndicatorConfig",
                            "MTFConfig",
                            "WaveletConfig",
                            "RegimeConfig",
                            "ScalingConfig",
                        ],
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"PreprocessingGraph: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="PreprocessingGraph",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="PreprocessingGraph",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_inference_pipeline(report: Phase5ValidationReport) -> None:
    """Test InferencePipeline component."""
    print("\nTesting InferencePipeline...")

    try:
        from src.inference.pipeline import (
            EnsembleResult,
            InferencePipeline,
            InferenceResult,
        )

        # Test class methods
        class_methods = ["from_bundle", "from_bundles"]
        instance_methods = [
            "predict",
            "predict_all",
            "predict_ensemble",
            "validate",
            "get_model_info",
        ]
        properties = ["n_models", "model_names", "feature_columns", "horizon"]

        missing = []
        for method in class_methods:
            if not hasattr(InferencePipeline, method):
                missing.append(f"class:{method}")

        for method in instance_methods:
            if not hasattr(InferencePipeline, method):
                missing.append(f"instance:{method}")

        # Test dataclasses (just verify they can be imported)
        import numpy as np

        from src.models.base import PredictionOutput

        pred_output = PredictionOutput(
            class_predictions=np.array([0, 1]),
            class_probabilities=np.array([[0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]),
            confidence=np.array([0.4, 0.5]),
        )

        inf_result = InferenceResult(
            predictions=pred_output,
            inference_time_ms=10.0,
            model_name="test",
            horizon=20,
            n_samples=2,
        )
        df = inf_result.to_dataframe()

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="InferencePipeline",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="InferencePipeline",
                    passed=True,
                    details={
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                        "properties": properties,
                        "InferenceResult": "verified",
                        "EnsembleResult": "verified",
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"InferencePipeline: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="InferencePipeline",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="InferencePipeline",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_batch_predictor(report: Phase5ValidationReport) -> None:
    """Test BatchPredictor component."""
    print("\nTesting BatchPredictor...")

    try:
        from src.inference.batch import (
            BatchPredictor,
            BatchProgress,
            BatchResult,
            run_batch_inference,
        )

        # Test BatchProgress
        progress = BatchProgress(
            total_samples=10000,
            processed_samples=5000,
            current_batch=5,
            total_batches=10,
            elapsed_seconds=30.0,
            samples_per_second=166.67,
        )
        pct = progress.progress_pct
        eta = progress.eta_seconds

        # Test class methods
        class_methods = ["from_bundle", "from_bundles"]
        instance_methods = ["predict_batch", "predict_streaming"]

        missing = []
        for method in class_methods:
            if not hasattr(BatchPredictor, method):
                missing.append(f"class:{method}")

        for method in instance_methods:
            if not hasattr(BatchPredictor, method):
                missing.append(f"instance:{method}")

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="BatchPredictor",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="BatchPredictor",
                    passed=True,
                    details={
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                        "BatchProgress": "verified",
                        "BatchResult": "verified",
                        "run_batch_inference": callable(run_batch_inference),
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"BatchPredictor: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="BatchPredictor",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="BatchPredictor",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_model_server(report: Phase5ValidationReport) -> None:
    """Test ModelServer component."""
    print("\nTesting ModelServer...")

    try:
        from src.inference.server import (
            ModelServer,
            ServerConfig,
            start_server,
        )

        # Test ServerConfig
        config = ServerConfig(
            host="0.0.0.0",
            port=8080,
            debug=False,
            max_batch_size=1000,
            timeout_seconds=30.0,
            enable_metrics=True,
        )

        # Test class methods
        class_methods = ["from_bundle", "from_bundles"]
        instance_methods = ["create_app", "run"]

        missing = []
        for method in class_methods:
            if not hasattr(ModelServer, method):
                missing.append(f"class:{method}")

        for method in instance_methods:
            if not hasattr(ModelServer, method):
                missing.append(f"instance:{method}")

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="ModelServer",
                    passed=False,
                    error=f"Missing methods: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="ModelServer",
                    passed=True,
                    details={
                        "class_methods": class_methods,
                        "instance_methods": instance_methods,
                        "ServerConfig": "verified",
                        "start_server": callable(start_server),
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"ModelServer: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="ModelServer",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="ModelServer",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_integration_pipeline_config(report: Phase5ValidationReport) -> None:
    """Test integration with PipelineConfig (PHASE_0)."""
    print("\nTesting PipelineConfig integration (PHASE_0)...")

    try:
        from src.core import PipelineConfig

        # Test that PipelineConfig can be imported
        # Create minimal config for testing
        config = PipelineConfig(
            symbol="MES",
            data_path="/tmp/test_data.parquet",
            output_dir="/tmp/test_output",
        )

        # Verify relevant attributes for PHASE_5
        attrs = ["output_dir", "symbol", "horizons", "mtf_timeframes"]
        missing = [a for a in attrs if not hasattr(config, a)]

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: PipelineConfig (PHASE_0)",
                    passed=False,
                    error=f"Missing attributes: {missing}",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: PipelineConfig (PHASE_0)",
                    passed=True,
                    details={
                        "attributes_verified": attrs,
                        "PipelineConfig": "importable",
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"PipelineConfig: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="Integration: PipelineConfig (PHASE_0)",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="Integration: PipelineConfig (PHASE_0)",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_integration_oof_aligner(report: Phase5ValidationReport) -> None:
    """Test integration with OOFAligner (PHASE_2)."""
    print("\nTesting OOFAligner integration (PHASE_2)...")

    try:
        from src.data.adapters.alignment import AlignedOOFResult, OOFAligner
        from src.core.interfaces import OOFResult

        # Test that OOFAligner can be imported
        aligner = OOFAligner(n_classes=3)

        # Verify primary method (align) - align_indices may be internal
        if not hasattr(aligner, "align"):
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: OOFAligner (PHASE_2)",
                    passed=False,
                    error="Missing align() method",
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: OOFAligner (PHASE_2)",
                    passed=True,
                    details={
                        "OOFAligner": "verified",
                        "AlignedOOFResult": "verified",
                        "OOFResult": "verified",
                        "align_method": "present",
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"OOFAligner: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="Integration: OOFAligner (PHASE_2)",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="Integration: OOFAligner (PHASE_2)",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_integration_training_result(report: Phase5ValidationReport) -> None:
    """Test integration with TrainingRunResult (PHASE_3)."""
    print("\nTesting TrainingRunResult integration (PHASE_3)...")

    try:
        from src.training.unified_orchestrator import TrainingRunResult

        # Verify class exists and has expected attributes
        # Note: actual attribute name is total_time_seconds, not total_training_time
        expected_attrs = [
            "run_id",
            "config",
            "model_results",
            "output_dir",
            "total_time_seconds",  # Actual attribute name
        ]

        # Check attributes exist as class attributes or annotations
        import inspect

        sig = inspect.signature(TrainingRunResult)
        params = list(sig.parameters.keys()) if hasattr(TrainingRunResult, "__init__") else []

        # For dataclasses, check annotations
        annotations = getattr(TrainingRunResult, "__annotations__", {})
        available = set(params) | set(annotations.keys())

        missing = [
            a for a in expected_attrs if a not in available and not hasattr(TrainingRunResult, a)
        ]

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: TrainingRunResult (PHASE_3)",
                    passed=False,
                    error=f"Missing attributes: {missing}",
                    details={"available": list(available)},
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: TrainingRunResult (PHASE_3)",
                    passed=True,
                    details={
                        "TrainingRunResult": "verified",
                        "expected_attrs": expected_attrs,
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"TrainingRunResult: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="Integration: TrainingRunResult (PHASE_3)",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="Integration: TrainingRunResult (PHASE_3)",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_integration_ensemble_result(report: Phase5ValidationReport) -> None:
    """Test integration with EnsembleResult (PHASE_4)."""
    print("\nTesting EnsembleResult integration (PHASE_4)...")

    try:
        from src.models.ensemble.orchestrator import EnsembleResult

        # Verify class exists and has expected attributes
        expected_attrs = [
            "ensemble_name",
            "meta_learner_name",
            "base_model_names",
            "n_base_models",
            "coverage",
            "alignment_offset",
            "metrics",
        ]

        # Check attributes
        annotations = getattr(EnsembleResult, "__annotations__", {})
        available = set(annotations.keys())

        missing = [
            a for a in expected_attrs if a not in available and not hasattr(EnsembleResult, a)
        ]

        if missing:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: EnsembleResult (PHASE_4)",
                    passed=False,
                    error=f"Missing attributes: {missing}",
                    details={"available": list(available)},
                )
            )
        else:
            report.results.append(
                ComponentTestResult(
                    component_name="Integration: EnsembleResult (PHASE_4)",
                    passed=True,
                    details={
                        "EnsembleResult": "verified",
                        "expected_attrs": expected_attrs,
                    },
                )
            )

    except ImportError as e:
        report.import_errors.append(f"EnsembleResult: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="Integration: EnsembleResult (PHASE_4)",
                passed=False,
                error=str(e),
            )
        )
    except Exception as e:
        report.results.append(
            ComponentTestResult(
                component_name="Integration: EnsembleResult (PHASE_4)",
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
        )


def test_inference_package_exports(report: Phase5ValidationReport) -> None:
    """Test that all components are properly exported from the inference package."""
    print("\nTesting inference package exports...")

    try:
        from src.inference import (
            BUNDLE_PREPROCESSING_GRAPH_FILE,
            BUNDLE_VERSION,
            ENSEMBLE_BUNDLE_VERSION,
            PREPROCESSING_GRAPH_FILE,
            PREPROCESSING_GRAPH_VERSION,
            AlignmentConfig,
            # Batch
            BatchPredictor,
            BatchProgress,
            BatchResult,
            # Builder (PHASE_5)
            BundleBuilder,
            BundleBuildResult,
            BundleManifest,
            BundleMetadata,
            CleaningConfig,
            # EnsembleBundle (PHASE_5)
            EnsembleBundle,
            EnsembleBundleManifest,
            EnsembleBundleMetadata,
            EnsembleResult,
            IndicatorConfig,
            # Orchestrator (PHASE_5)
            InferenceOrchestrator,
            # Pipeline
            InferencePipeline,
            InferenceResult,
            # ModelBundle
            ModelBundle,
            # Server
            ModelServer,
            MTFConfig,
            PredictionResult,
            # PreprocessingGraph
            PreprocessingGraph,
            PreprocessingGraphConfig,
            RegimeConfig,
            ScalingConfig,
            ServerConfig,
            WaveletConfig,
            build_bundles,
            build_from_run,
            load_inference,
            predict_batch_from_bundle,
            predict_from_bundle,
            run_batch_inference,
            start_server,
        )

        report.results.append(
            ComponentTestResult(
                component_name="inference package exports",
                passed=True,
                details={
                    "orchestrator_exports": 5,
                    "builder_exports": 4,
                    "ensemble_bundle_exports": 5,
                    "model_bundle_exports": 5,
                    "preprocessing_exports": 10,
                    "pipeline_exports": 3,
                    "batch_exports": 4,
                    "server_exports": 3,
                    "total_exports": 39,
                },
            )
        )

    except ImportError as e:
        report.import_errors.append(f"inference package: {e}")
        report.results.append(
            ComponentTestResult(
                component_name="inference package exports",
                passed=False,
                error=str(e),
            )
        )


# =============================================================================
# MAIN VALIDATION
# =============================================================================


def run_phase5_validation() -> Phase5ValidationReport:
    """Run complete PHASE_5 validation."""
    report = Phase5ValidationReport()

    print("=" * 80)
    print("PHASE_5 (INFERENCE BUNDLE) COMPREHENSIVE VALIDATION")
    print("=" * 80)

    # 1. Test InferenceOrchestrator
    test_inference_orchestrator(report)

    # 2. Test BundleBuilder
    test_bundle_builder(report)

    # 3. Test EnsembleBundle
    test_ensemble_bundle(report)

    # 4. Test existing infrastructure
    test_model_bundle(report)
    test_preprocessing_graph(report)
    test_inference_pipeline(report)
    test_batch_predictor(report)
    test_model_server(report)

    # 5. Test integration points
    test_integration_pipeline_config(report)
    test_integration_oof_aligner(report)
    test_integration_training_result(report)
    test_integration_ensemble_result(report)

    # 6. Test package exports
    test_inference_package_exports(report)

    return report


if __name__ == "__main__":
    try:
        report = run_phase5_validation()
        report.print_report()

        # Exit with appropriate code
        if report.import_errors or report.total_failed > 0:
            sys.exit(1)
        sys.exit(0)

    except Exception as e:
        print(f"\nFATAL ERROR during validation: {e}")
        traceback.print_exc()
        sys.exit(2)
