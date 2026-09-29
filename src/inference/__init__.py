"""
Inference package for ML Model Factory.

Production entry point: ``load_deploy_artifact`` loads whatever
``MLFactory.run()`` deployed for a horizon (model, ensemble, regime or
meta-labeling bundle) and predicts straight from raw OHLCV bars, replaying the
training feature spec, scaler and calibrator:

    from src.inference import load_deploy_artifact

    artifact = load_deploy_artifact(result.deploy_path, horizon=5)
    pred = artifact.predict_from_raw(raw_ohlcv_df)

``UniversalInferencePipeline`` is the multi-bundle interface over the same
bundles (mixed 2D/3D/4D models plus an optional ensemble):

    from src.inference import UniversalInferencePipeline

    pipeline = UniversalInferencePipeline.from_bundles(
        ["./bundles/xgboost_h5", "./bundles/lstm_h5"],
        ensemble_path="./bundles/ensemble_h5",
    )
    per_model = pipeline.predict_from_raw(raw_ohlcv_df, bundle_index=0)
    combined = pipeline.predict_ensemble(raw_ohlcv_df)

Building blocks:
- ModelBundle / EnsembleBundle / RegimeBundle / MetaLabelingBundle: serializable artifacts
- BundleBuilder / build_bundles: create bundles from a TrainingRunResult
- PreprocessingGraph: serializable preprocessing for train/serve parity
"""

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
    describe_bundle,
    load_bundle,
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

# NOTE: these are FIRST-PARTY modules — no except-ImportError guards. A
# guard here would silently mask refactoring bugs by exporting None.
from src.inference.meta_labeling_bundle import (
    META_LABELING_BUNDLE_VERSION,
    MetaLabelingBundle,
    MetaLabelingPrediction,
)
from src.inference.preprocessing_graph import (
    PREPROCESSING_GRAPH_FILE,
    PREPROCESSING_GRAPH_VERSION,
    PreprocessingGraph,
    PreprocessingGraphConfig,
)
from src.inference.regime_bundle import (
    REGIME_BUNDLE_VERSION,
    RegimeBundle,
)
from src.inference.universal_pipeline import (
    UniversalInferencePipeline,
    UniversalPredictionResult,
)

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
    "PREPROCESSING_GRAPH_VERSION",
    "PREPROCESSING_GRAPH_FILE",
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
    # Deploy artifact
    "DeployManifest",
    "HorizonArtifactEntry",
    "HorizonManifest",
    "DEPLOY_MANIFEST_FILE",
    "DEPLOY_VERSION",
    "load_deploy_artifact",
    "load_bundle",
    "describe_bundle",
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
    "RegimeBundle",
    "REGIME_BUNDLE_VERSION",
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "META_LABELING_BUNDLE_VERSION",
]
