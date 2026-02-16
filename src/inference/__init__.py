"""
Inference package for ML Model Factory.

This package provides end-to-end inference capabilities:
- InferenceOrchestrator: THE single entry point for all inference (PHASE_5)
- ModelBundle: Serializable container for model artifacts
- PreprocessingGraph: Serializable preprocessing pipeline for train/serve parity
- BundleBuilder: Create bundles from PHASE_3/PHASE_4 training results
- InferencePipeline: High-level prediction interface
- BatchPredictor: Efficient batch processing
- ModelServer: Optional HTTP serving

Usage:
    # RECOMMENDED: Use InferenceOrchestrator (PHASE_5 unified interface)
    from src.core import PipelineConfig
    from src.inference import InferenceOrchestrator

    config = PipelineConfig.load("./experiments/exp_001/config.json")
    orchestrator = InferenceOrchestrator.from_experiment(config)
    result = orchestrator.predict(X_new)

    # Or load from specific bundle
    orchestrator = InferenceOrchestrator.from_bundle("./bundles/xgb_h20")
    result = orchestrator.predict(X_new)

    # End-to-end prediction from raw OHLCV
    result = orchestrator.predict_from_raw(raw_ohlcv_df)

    # Batch inference with orchestrator
    predictions_df = orchestrator.predict_batch(data, output_path="predictions.parquet")

    # Bundle a trained model manually
    from src.inference import ModelBundle

    bundle = ModelBundle.from_training(
        model=trained_model,
        scaler=fitted_scaler,
        feature_columns=feature_cols,
        horizon=20,
    )
    bundle.save("./bundles/xgb_h20")

    # Build bundles from PHASE_3 TrainingRunResult
    from src.inference import BundleBuilder, build_bundles

    config = PipelineConfig(...)
    builder = BundleBuilder(config)
    result = builder.build_from_training_result(training_result)

    # Or use convenience function
    result = build_bundles(config, training_result, ensemble_result)

    # Load and predict (lower-level interface)
    from src.inference import InferencePipeline

    pipeline = InferencePipeline.from_bundle("./bundles/xgb_h20")
    result = pipeline.predict(X_new)

    # Batch inference (lower-level interface)
    from src.inference import BatchPredictor

    predictor = BatchPredictor.from_bundle("./bundles/xgb_h20")
    result = predictor.predict_batch(df, output_path="predictions.parquet")

    # With preprocessing graph for raw OHLCV inference
    from src.inference import PreprocessingGraph

    graph = PreprocessingGraph.from_pipeline_config(pipeline_config)
    bundle.set_preprocessing_graph(graph)
    bundle.save("./bundles/xgb_h20_with_graph")

    # At inference time - predict directly from raw OHLCV
    bundle = ModelBundle.load("./bundles/xgb_h20_with_graph")
    predictions = bundle.predict_from_raw(raw_ohlcv_df)
"""

from src.inference.batch import (
    BatchInference,
    BatchInferenceResult,
    BatchPredictor,
    BatchProgress,
    BatchResult,
    ModelPrediction,
    run_batch_inference,
)
from src.inference.builder import (
    BundleBuilder,
    BundleBuildResult,
    build_bundles,
    build_from_run,
)
from src.inference.bundle import (
    BUNDLE_FEATURE_SPEC_FILE,
    BUNDLE_PREPROCESSING_GRAPH_FILE,
    BUNDLE_VERSION,
    BundleManifest,
    BundleMetadata,
    ModelBundle,
)

# Deploy artifact packaging
from src.inference.deploy import (
    DEPLOY_MANIFEST_FILE,
    DEPLOY_VERSION,
    DeployManifest,
    HorizonArtifactEntry,
    HorizonManifest,
    load_deploy_artifact,
    select_deploy_artifact,
    validate_deploy_artifact,
)

# PHASE_5: Ensemble bundle for stacking ensembles
from src.inference.ensemble_bundle import (
    ENSEMBLE_BUNDLE_VERSION,
    AlignmentConfig,
    EnsembleBundle,
    EnsembleBundleManifest,
    EnsembleBundleMetadata,
)

# Inference errors
from src.inference.errors import (
    AdapterRoutingError,
    InferenceError,
    PreprocessingError,
    ShapeMismatchError,
)

# PHASE_5: InferenceOrchestrator - THE single entry point for inference
from src.inference.orchestrator import (
    InferenceOrchestrator,
    PredictionResult,
    load_inference,
    predict_batch_from_bundle,
    predict_from_bundle,
)
from src.inference.pipeline import (
    EnsembleResult,
    InferencePipeline,
    InferenceResult,
)
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
from src.inference.server import (
    ModelServer,
    ServerConfig,
    start_server,
)

# UniversalInferencePipeline - THE single entry point for all inference
try:
    from src.inference.universal_pipeline import (
        UniversalInferencePipeline,
        UniversalPredictionResult,
    )
except ImportError:
    UniversalInferencePipeline = None  # type: ignore[assignment,misc]
    UniversalPredictionResult = None  # type: ignore[assignment,misc]

# Special mode bundles
try:
    from src.inference.walk_forward_bundle import (
        WALK_FORWARD_BUNDLE_VERSION,
        WalkForwardBundle,
        WalkForwardMetadata,
    )
except ImportError:
    WalkForwardBundle = None  # type: ignore[assignment,misc]
    WalkForwardMetadata = None  # type: ignore[assignment,misc]
    WALK_FORWARD_BUNDLE_VERSION = "0.0.0"

try:
    from src.inference.regime_bundle import (
        REGIME_BUNDLE_VERSION,
        RegimeBundle,
    )
except ImportError:
    RegimeBundle = None  # type: ignore[assignment,misc]
    REGIME_BUNDLE_VERSION = "0.0.0"

try:
    from src.inference.regime_detector import (
        REGIME_DETECTOR_VERSION,
        RegimeDetector,
        RegimeDetectorConfig,
    )
except ImportError:
    RegimeDetector = None  # type: ignore[assignment,misc]
    RegimeDetectorConfig = None  # type: ignore[assignment,misc]
    REGIME_DETECTOR_VERSION = "0.0.0"

try:
    from src.inference.meta_labeling_bundle import (
        META_LABELING_BUNDLE_VERSION,
        MetaLabelingBundle,
        MetaLabelingBundleMetadata,
        MetaLabelingPrediction,
    )
except ImportError:
    MetaLabelingBundle = None  # type: ignore[assignment,misc]
    MetaLabelingPrediction = None  # type: ignore[assignment,misc]
    MetaLabelingBundleMetadata = None  # type: ignore[assignment,misc]
    META_LABELING_BUNDLE_VERSION = "0.0.0"

__all__ = [
    # Bundle
    "ModelBundle",
    "BundleMetadata",
    "BundleManifest",
    "BUNDLE_VERSION",
    "BUNDLE_PREPROCESSING_GRAPH_FILE",
    "BUNDLE_FEATURE_SPEC_FILE",
    # Preprocessing Graph
    "PreprocessingGraph",
    "PreprocessingGraphConfig",
    "CleaningConfig",
    "IndicatorConfig",
    "MTFConfig",
    "WaveletConfig",
    "RegimeConfig",
    "ScalingConfig",
    "PREPROCESSING_GRAPH_VERSION",
    "PREPROCESSING_GRAPH_FILE",
    # Pipeline
    "InferencePipeline",
    "InferenceResult",
    "EnsembleResult",
    # Batch data processing
    "BatchPredictor",
    "BatchProgress",
    "BatchResult",
    "run_batch_inference",
    # Parallel ensemble inference
    "BatchInference",
    "BatchInferenceResult",
    "ModelPrediction",
    # Server
    "ModelServer",
    "ServerConfig",
    "start_server",
    # Builder (PHASE_5)
    "BundleBuilder",
    "BundleBuildResult",
    "build_bundles",
    "build_from_run",
    # Ensemble Bundle (PHASE_5)
    "EnsembleBundle",
    "EnsembleBundleMetadata",
    "EnsembleBundleManifest",
    "AlignmentConfig",
    "ENSEMBLE_BUNDLE_VERSION",
    # InferenceOrchestrator (PHASE_5) - THE single entry point
    "InferenceOrchestrator",
    "PredictionResult",
    "load_inference",
    "predict_from_bundle",
    "predict_batch_from_bundle",
    # Deploy artifact
    "DeployManifest",
    "HorizonArtifactEntry",
    "HorizonManifest",
    "DEPLOY_MANIFEST_FILE",
    "DEPLOY_VERSION",
    "load_deploy_artifact",
    "select_deploy_artifact",
    "validate_deploy_artifact",
    # Inference errors
    "InferenceError",
    "ShapeMismatchError",
    "AdapterRoutingError",
    "PreprocessingError",
    # UniversalInferencePipeline
    "UniversalInferencePipeline",
    "UniversalPredictionResult",
    # Special mode bundles
    "WalkForwardBundle",
    "WalkForwardMetadata",
    "WALK_FORWARD_BUNDLE_VERSION",
    "RegimeBundle",
    "REGIME_BUNDLE_VERSION",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "REGIME_DETECTOR_VERSION",
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "MetaLabelingBundleMetadata",
    "META_LABELING_BUNDLE_VERSION",
]
