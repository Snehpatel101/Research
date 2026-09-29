"""
BundleBuilder - Create inference bundles from training results.

Bridges PHASE_3 training and PHASE_4 ensemble with PHASE_5 inference.
Uses PipelineConfig as the single source of truth.

This module provides:
- BundleBuilder: Main class for building inference bundles
- BundleBuildResult: Result container with bundle paths and metadata
- build_bundles: Convenience function for quick bundle creation

Example:
    from src.core import PipelineConfig
    from src.inference import BundleBuilder

    config = PipelineConfig(...)
    builder = BundleBuilder(config)

    # Build from PHASE_3 training result
    result = builder.build_from_training_result(training_result)

    # Build from PHASE_4 ensemble result
    result = builder.build_ensemble_bundle(ensemble_result)

    # Or build all at once
    result = builder.build_all(
        training_result=training_result,
        ensemble_result=ensemble_result,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core import PipelineConfig

if TYPE_CHECKING:
    from src.models.ensemble.orchestrator import EnsembleResult
    from src.models.training.unified_orchestrator import TrainingRunResult

logger = logging.getLogger(__name__)


def _get_trainer_protocol() -> type | None:
    """Lazy import TrainerProtocol to avoid circular imports."""
    try:
        from src.core.protocols import TrainerProtocol

        return TrainerProtocol
    except ImportError:
        return None


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class BundleBuildResult:
    """
    Result from bundle building.

    Attributes:
        bundle_paths: List of paths to created base model bundles
        ensemble_bundle_path: Optional path to ensemble bundle
        n_bundles: Total number of bundles created
        total_size_mb: Total size of all bundles in MB
        metadata: Additional metadata about the build
    """

    bundle_paths: list[Path] = field(default_factory=list)
    ensemble_bundle_path: Path | None = None
    n_bundles: int = 0
    total_size_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bundle_paths": [str(p) for p in self.bundle_paths],
            "ensemble_bundle_path": (
                str(self.ensemble_bundle_path) if self.ensemble_bundle_path else None
            ),
            "n_bundles": self.n_bundles,
            "total_size_mb": self.total_size_mb,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """
        Save build result to JSON.

        Args:
            path: Path to save the result

        Returns:
            Path to saved file
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BundleBuildResult:
        """
        Load build result from JSON.

        Args:
            path: Path to the saved result

        Returns:
            Loaded BundleBuildResult
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            bundle_paths=[Path(p) for p in data.get("bundle_paths", [])],
            ensemble_bundle_path=(
                Path(data["ensemble_bundle_path"]) if data.get("ensemble_bundle_path") else None
            ),
            n_bundles=data.get("n_bundles", 0),
            total_size_mb=data.get("total_size_mb", 0.0),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "BundleBuildResult:",
            f"  Base bundles: {len(self.bundle_paths)}",
            f"  Ensemble bundle: {'Yes' if self.ensemble_bundle_path else 'No'}",
            f"  Total bundles: {self.n_bundles}",
            f"  Total size: {self.total_size_mb:.2f} MB",
        ]
        if self.bundle_paths:
            lines.append("  Bundle paths:")
            for p in self.bundle_paths[:5]:
                lines.append(f"    - {p.name}")
            if len(self.bundle_paths) > 5:
                lines.append(f"    ... and {len(self.bundle_paths) - 5} more")
        return "\n".join(lines)


# =============================================================================
# BUNDLE BUILDER
# =============================================================================


class BundleBuilder:
    """
    Create inference bundles from training results.

    Integrates PHASE_3 training output with PHASE_5 inference bundles.
    Uses PipelineConfig as the single source of truth.

    The builder handles:
    - Extracting trained models from TrainingRunResult
    - Creating PreprocessingGraph for train/serve parity
    - Building ModelBundle for each trained model
    - Building EnsembleBundle from EnsembleResult
    - Computing and storing bundle metadata

    Usage:
        from src.core import PipelineConfig
        from src.inference import BundleBuilder

        config = PipelineConfig(...)
        builder = BundleBuilder(config)

        # Build from PHASE_3 training result
        result = builder.build_from_training_result(training_result)

        # Build from PHASE_4 ensemble result
        result = builder.build_ensemble_bundle(ensemble_result)

        # Or build all at once
        result = builder.build_all(
            training_result=training_result,
            ensemble_result=ensemble_result,
        )
    """

    def __init__(
        self, config: PipelineConfig, feature_pipeline: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize BundleBuilder.

        Args:
            config: PipelineConfig instance - THE single source of truth
            feature_pipeline: Raw-OHLCV -> features recipe recorded by training
                (``{"bar_timeframe": ..., "engineer": FeatureEngineer.to_spec()}``).
                Required for bundles that can predict from raw OHLCV.
        """
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.output_dir = Path(config.output_dir)
        self.bundles_dir = self.output_dir / "bundles"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

        logger.info("BundleBuilder initialized")
        logger.info(f"  Output: {self.bundles_dir}")

    @classmethod
    def from_config(cls, config: PipelineConfig) -> BundleBuilder:
        """
        Create builder from PipelineConfig.

        Args:
            config: PipelineConfig instance

        Returns:
            BundleBuilder instance
        """
        return cls(config)

    @classmethod
    def from_training_run(cls, run_path: str | Path) -> BundleBuilder:
        """
        Create builder from a completed training run.

        Loads PipelineConfig from the run directory.

        Args:
            run_path: Path to training run directory containing config.json

        Returns:
            BundleBuilder instance

        Raises:
            FileNotFoundError: If config.json not found in run_path
        """
        run_path = Path(run_path)
        config_path = run_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {config_path}. "
                "Ensure the training run saved its configuration."
            )

        config = PipelineConfig.load(config_path)
        return cls(config)

    def build_from_training_result(
        self,
        training_result: TrainingRunResult,
        include_preprocessing_graph: bool = True,
        include_calibrator: bool = True,
        feature_specs: dict[str, Any] | None = None,
    ) -> BundleBuildResult:
        """
        Build bundles from PHASE_3 TrainingRunResult.

        Creates one bundle per trained model, each containing:
        - The trained model
        - Feature scaler
        - Feature column names
        - Preprocessing graph (optional, for raw OHLCV inference)
        - Probability calibrator (optional)
        - Feature spec (optional, for 5-dimension optimization parity)
        - Training metrics

        Args:
            training_result: Result from UnifiedTrainingOrchestrator.train()
            include_preprocessing_graph: Include preprocessing graph for raw inference
            include_calibrator: Include probability calibrator if available
            feature_specs: Optional dict mapping model_key -> FeatureSpec for each model.
                          When provided, the bundle will include the FeatureSpec to ensure
                          inference uses the exact same configuration as training.

        Returns:
            BundleBuildResult with paths to created bundles
        """
        from src.inference.preprocessing_graph import PreprocessingGraph

        bundle_paths: list[Path] = []
        build_metadata: dict[str, Any] = {
            "run_id": training_result.run_id,
            "models_bundled": [],
            "models_skipped": [],
        }

        # Create preprocessing graph if needed (restricted per model below)
        base_graph: PreprocessingGraph | None = None
        if include_preprocessing_graph:
            base_graph = self._create_preprocessing_graph()

        # Build bundle for each model
        for key, model_result in training_result.model_results.items():
            model_name = model_result.model_name
            horizon = model_result.horizon
            kind = model_result.mode_artifacts.get("kind", "model")
            logger.info(f"Building {kind} bundle for {key}...")

            def make(trainer: Any, name: str, key: str = key, mr: Any = model_result) -> Any:
                return self._make_model_bundle(
                    key=key,
                    model_result=mr,
                    model_name=name,
                    trainer=trainer,
                    base_graph=base_graph,
                    include_calibrator=include_calibrator,
                    feature_specs=feature_specs,
                    run_id=training_result.run_id,
                )

            try:
                if kind == "regime":
                    from src.inference.regime_bundle import RegimeBundle

                    art = model_result.mode_artifacts
                    bundle: Any = RegimeBundle(
                        regime_bundles={
                            regime: make(trainer, model_name)
                            for regime, trainer in art["regime_trainers"].items()
                        },
                        detector_config=art["detector_config"],
                        default_regime=art["default_regime"],
                        model_name=model_name,
                        horizon=horizon,
                        metrics=model_result.metrics,
                    )
                elif kind == "meta_labeling":
                    from src.inference.meta_labeling_bundle import MetaLabelingBundle

                    art = model_result.mode_artifacts
                    bundle = MetaLabelingBundle(
                        primary_bundle=make(model_result.trainer, art["primary_model"]),
                        meta_model=art["meta_model"],
                        threshold=art["threshold"],
                        meta_model_name=art["meta_model_name"],
                        horizon=horizon,
                        metrics=model_result.metrics,
                    )
                else:
                    if model_result.trainer is None:
                        logger.warning(f"No trainer for {key}, skipping bundle")
                        build_metadata["models_skipped"].append(key)
                        continue
                    bundle = make(model_result.trainer, model_name)

                bundle_path = self.bundles_dir / f"{model_name}_h{horizon}"
                bundle.save(bundle_path, overwrite=True)
                bundle_paths.append(bundle_path)
                build_metadata["models_bundled"].append(
                    {
                        "key": key,
                        "kind": kind,
                        "model_name": model_name,
                        "horizon": horizon,
                        "bundle_path": str(bundle_path),
                    }
                )
                logger.info(f"Built bundle: {bundle_path}")

            except Exception as e:
                logger.error(f"Failed to build bundle for {key}: {e}")
                build_metadata["models_skipped"].append(key)
                continue

        # Calculate total size
        total_size = self._calculate_total_size(bundle_paths)

        return BundleBuildResult(
            bundle_paths=bundle_paths,
            n_bundles=len(bundle_paths),
            total_size_mb=total_size,
            metadata=build_metadata,
        )

    def _make_model_bundle(
        self,
        key: str,
        model_result: Any,
        model_name: str,
        trainer: Any,
        base_graph: Any,
        include_calibrator: bool,
        feature_specs: dict[str, Any] | None,
        run_id: str,
    ) -> Any:
        """Assemble (without saving) the ModelBundle for one trained model."""
        from src.inference.bundle import ModelBundle

        model = self._extract_model(trainer)
        if model is None:
            raise ValueError(f"No model in trainer for {key}")

        scaler = self._extract_scaler(trainer)
        feature_columns = self._extract_feature_columns(trainer, model_result.n_features)

        # Propagate feature names to the model if it supports it
        if hasattr(model, "set_feature_names") and callable(model.set_feature_names):
            model.set_feature_names(feature_columns)

        calibrator = None
        if include_calibrator:
            calibrator = self._extract_calibrator(trainer)
            # Also check model_result (calibrator propagated from orchestrator)
            if calibrator is None and getattr(model_result, "calibrator", None) is not None:
                calibrator = model_result.calibrator

        # Get feature spec: explicit dict, or auto-generate as fallback
        feature_spec = feature_specs.get(key) if feature_specs is not None else None
        if feature_spec is None:
            feature_spec = self._auto_generate_feature_spec(
                model_name=model_name,
                feature_columns=feature_columns,
                horizon=model_result.horizon,
                model_result=model_result,
            )

        return ModelBundle.from_training(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            horizon=model_result.horizon,
            calibrator=calibrator,
            preprocessing_graph=(
                base_graph.with_feature_columns(feature_columns) if base_graph else None
            ),
            feature_spec=feature_spec,
            symbol=self.config.symbol,
            training_metrics=model_result.metrics,
            extra_metadata={
                "training_run_id": run_id,
                "training_time_seconds": model_result.training_time_seconds,
                "data_rank": model_result.data_rank,
                **self._extract_timeframe_metadata(trainer),
            },
            model_name=model_name,
        )

    def build_ensemble_from_training_result(
        self,
        training_result: TrainingRunResult,
        base_bundles: list[Path],
    ) -> Path | None:
        """
        Build the stacking-ensemble bundle for a training run, if one was trained.

        Args:
            training_result: Result whose ``ensemble_result`` holds the fitted
                meta-learner (as ``trainer``) and whose ``aligned_oof`` defines
                the stacking layout.
            base_bundles: Paths of the base-model bundles the ensemble combines.

        Returns:
            Path to the ensemble bundle, or None when no ensemble was trained.
        """
        from src.models.ensemble.orchestrator import EnsembleResult

        trained = training_result.ensemble_result
        aligned = training_result.aligned_oof
        if trained is None or trained.trainer is None or aligned is None:
            return None

        # OOF keys carry the horizon ("xgboost_h5"); bundles are keyed by model name
        suffix = f"_h{trained.horizon}"
        base_model_names = [name.removesuffix(suffix) for name in aligned.model_names]
        coverage = aligned.coverage
        ensemble_result = EnsembleResult(
            ensemble_name=f"{self.config.meta_learner}_ensemble",
            meta_learner_name=self.config.meta_learner,
            base_model_names=base_model_names,
            metrics={k: float(v) for k, v in trained.metrics.items() if isinstance(v, int | float)},
            stacking_dataset=training_result.stacking_dataset,
            aligned_oof=aligned,
            training_time_seconds=trained.training_time_seconds,
            n_base_models=len(base_model_names),
            coverage=min(coverage.values()) if isinstance(coverage, dict) else float(coverage),
            meta_learner=trained.trainer,
        )
        by_name = {p.name.removesuffix(suffix): p for p in base_bundles}
        missing = [n for n in base_model_names if n not in by_name]
        if missing:
            raise ValueError(f"Ensemble base models without bundles: {missing}")
        return self.build_ensemble_bundle(
            ensemble_result, base_bundles=[by_name[n] for n in base_model_names]
        )

    def build_ensemble_bundle(
        self,
        ensemble_result: EnsembleResult,
        base_bundles: list[Path] | None = None,
    ) -> Path:
        """
        Build ensemble bundle from PHASE_4 EnsembleResult.

        Creates a bundle directory compatible with EnsembleBundle.load(),
        containing:
        - manifest.json (file listing with checksums)
        - metadata.json (EnsembleBundleMetadata format)
        - base_bundles.json (model name -> relative path mapping)
        - stacking_features.json (feature column names)
        - alignment_config.json (OOF alignment configuration)
        - meta_learner/ (serialized meta-learner model)

        Args:
            ensemble_result: Result from EnsembleOrchestrator.train()
            base_bundles: Optional paths to base model bundles

        Returns:
            Path to ensemble bundle

        Raises:
            ValueError: If ensemble_result is invalid
        """
        from src.inference.ensemble_bundle import (
            ENSEMBLE_ALIGNMENT_CONFIG_FILE,
            ENSEMBLE_BASE_BUNDLES_FILE,
            ENSEMBLE_BUNDLE_VERSION,
            ENSEMBLE_MANIFEST_FILE,
            ENSEMBLE_META_LEARNER_DIR,
            ENSEMBLE_METADATA_FILE,
            ENSEMBLE_STACKING_FEATURES_FILE,
        )

        ensemble_dir = self.bundles_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        files: list[str] = []
        base_bundles = base_bundles or []

        # --- Extract stacking feature names from aligned OOF ---
        stacking_feature_names: list[str] = []
        if ensemble_result.aligned_oof is not None:
            aligned = ensemble_result.aligned_oof
            if hasattr(aligned, "get_feature_names"):
                stacking_feature_names = aligned.get_feature_names()
        if not stacking_feature_names and ensemble_result.stacking_dataset is not None:
            stacking_data = ensemble_result.stacking_dataset.data
            stacking_feature_names = [
                c for c in stacking_data.columns if c not in ("y_true", "datetime")
            ]

        # --- 1. metadata.json (EnsembleBundleMetadata format) ---
        horizon = self.config.horizons[0] if self.config.horizons else 20
        metadata = {
            "version": ENSEMBLE_BUNDLE_VERSION,
            "created_at": datetime.now().isoformat(),
            "meta_learner_name": ensemble_result.meta_learner_name,
            "base_model_names": ensemble_result.base_model_names,
            "horizon": horizon,
            "n_base_models": ensemble_result.n_base_models,
            "n_stacking_features": len(stacking_feature_names),
            "symbol": self.config.symbol,
            "coverage": ensemble_result.coverage,
            "alignment_offset": ensemble_result.alignment_offset,
            "metrics": ensemble_result.metrics,
            "extra": {
                "training_time_seconds": ensemble_result.training_time_seconds,
                "ensemble_name": ensemble_result.ensemble_name,
            },
        }
        metadata_path = ensemble_dir / ENSEMBLE_METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        files.append(ENSEMBLE_METADATA_FILE)

        # --- 2. stacking_features.json ---
        stacking_path = ensemble_dir / ENSEMBLE_STACKING_FEATURES_FILE
        with open(stacking_path, "w") as f:
            json.dump(
                {
                    "feature_names": stacking_feature_names,
                    "n_features": len(stacking_feature_names),
                },
                f,
                indent=2,
            )
        files.append(ENSEMBLE_STACKING_FEATURES_FILE)

        # --- 3. base_bundles.json (paths + model_names) ---
        relative_paths: list[str] = []
        for p in base_bundles:
            try:
                relative_paths.append(str(Path(p).relative_to(ensemble_dir.parent)))
            except ValueError:
                relative_paths.append(str(p))
        bundles_path = ensemble_dir / ENSEMBLE_BASE_BUNDLES_FILE
        with open(bundles_path, "w") as f:
            json.dump(
                {
                    "paths": relative_paths,
                    "model_names": ensemble_result.base_model_names,
                },
                f,
                indent=2,
            )
        files.append(ENSEMBLE_BASE_BUNDLES_FILE)

        # --- 4. alignment_config.json ---
        alignment_data: dict[str, Any] = {
            "strategy": "intersection",
            "n_classes": 3,
            "model_offsets": {},
            "sequence_lengths": {},
        }
        if ensemble_result.aligned_oof is not None:
            aligned = ensemble_result.aligned_oof
            alignment_data["n_classes"] = getattr(aligned, "n_classes", 3)
            if hasattr(aligned, "coverage"):
                alignment_data["model_offsets"] = dict.fromkeys(ensemble_result.base_model_names, 0)
        alignment_path = ensemble_dir / ENSEMBLE_ALIGNMENT_CONFIG_FILE
        with open(alignment_path, "w") as f:
            json.dump(alignment_data, f, indent=2)
        files.append(ENSEMBLE_ALIGNMENT_CONFIG_FILE)

        # --- 5. meta_learner/ (serialized with the meta-learner's own save) ---
        if ensemble_result.meta_learner is None:
            raise ValueError("EnsembleResult has no fitted meta_learner to bundle")
        ensemble_result.meta_learner.save(ensemble_dir / ENSEMBLE_META_LEARNER_DIR)
        files.append(ENSEMBLE_META_LEARNER_DIR)

        # --- 6. manifest.json (file listing) ---
        manifest = {
            "version": ENSEMBLE_BUNDLE_VERSION,
            "files": files,
            "checksums": {},
        }
        manifest_path = ensemble_dir / ENSEMBLE_MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Built ensemble bundle: {ensemble_dir}")

        return ensemble_dir

    def build_all(
        self,
        training_result: TrainingRunResult | None = None,
        ensemble_result: EnsembleResult | None = None,
        include_preprocessing_graph: bool = True,
        feature_specs: dict[str, Any] | None = None,
    ) -> BundleBuildResult:
        """
        Build all bundles from training and ensemble results.

        Convenience method that combines build_from_training_result and
        build_ensemble_bundle into a single call.

        Args:
            training_result: PHASE_3 training result (optional)
            ensemble_result: PHASE_4 ensemble result (optional)
            include_preprocessing_graph: Include preprocessing graph
            feature_specs: Optional dict mapping model_key -> FeatureSpec

        Returns:
            BundleBuildResult with all bundle paths

        Raises:
            ValueError: If both training_result and ensemble_result are None
        """
        if training_result is None and ensemble_result is None:
            raise ValueError("At least one of training_result or ensemble_result must be provided")

        bundle_paths: list[Path] = []
        ensemble_path: Path | None = None
        metadata: dict[str, Any] = {}

        # Build base model bundles
        if training_result is not None:
            base_result = self.build_from_training_result(
                training_result,
                include_preprocessing_graph=include_preprocessing_graph,
                feature_specs=feature_specs,
            )
            bundle_paths.extend(base_result.bundle_paths)
            metadata["base_build"] = base_result.metadata

        # Build ensemble bundle
        if ensemble_result is not None:
            ensemble_path = self.build_ensemble_bundle(
                ensemble_result,
                base_bundles=bundle_paths,
            )
            metadata["ensemble_build"] = {
                "ensemble_name": ensemble_result.ensemble_name,
                "n_base_models": ensemble_result.n_base_models,
            }

        # Calculate total size
        all_paths = bundle_paths + ([ensemble_path] if ensemble_path else [])
        total_size = self._calculate_total_size(all_paths)

        # Save build result
        build_result = BundleBuildResult(
            bundle_paths=bundle_paths,
            ensemble_bundle_path=ensemble_path,
            n_bundles=len(bundle_paths) + (1 if ensemble_path else 0),
            total_size_mb=total_size,
            metadata=metadata,
        )

        # Save build result to disk
        result_path = self.bundles_dir / "build_result.json"
        build_result.save(result_path)
        logger.info(f"Saved build result to {result_path}")

        return build_result

    def _create_preprocessing_graph(self) -> Any:
        """
        Create the raw-OHLCV preprocessing graph from the recorded feature pipeline.

        Returns:
            PreprocessingGraph, or None when training recorded no feature
            pipeline (bundles then only accept pre-computed features).
        """
        from src.inference.preprocessing_graph import PreprocessingGraph

        if self.feature_pipeline is None:
            logger.warning(
                "No feature pipeline recorded; bundles will not support predict_from_raw()"
            )
            return None

        return PreprocessingGraph.from_feature_pipeline(
            self.feature_pipeline,
            symbol=self.config.symbol,
            horizon=self.config.horizons[0] if self.config.horizons else 20,
        )

    @staticmethod
    def _extract_timeframe_metadata(trainer: Any) -> dict[str, Any]:
        """Timeframes a 4D multi-stream model was trained on (anchor first).

        Inference must rebuild the exact same streams; the bundle cannot
        re-derive them from config because the adapter resolves them from
        the model contract and the data actually available.
        """
        timeframe_names = list(getattr(trainer, "timeframe_names", None) or [])
        if not timeframe_names:
            return {}
        return {
            "timeframe_names": timeframe_names,
            "mtf_timeframes": timeframe_names[1:],
        }

    def _extract_model(self, trainer: Any) -> Any | None:
        """
        Extract model from trainer instance.

        Uses TrainerProtocol when available, falls back to duck-typing
        for legacy trainer implementations.

        Args:
            trainer: Trainer instance

        Returns:
            Model instance or None
        """
        # Protocol-aware extraction (preferred)
        proto = _get_trainer_protocol()
        if proto is not None and isinstance(trainer, proto):
            return trainer.model

        # Legacy duck-typing fallback
        logger.debug("Trainer does not satisfy TrainerProtocol, using duck-typing fallback")
        for attr in ["model", "_model", "estimator", "_estimator"]:
            model = getattr(trainer, attr, None)
            if model is not None:
                return model

        if hasattr(trainer, "get_model") and callable(trainer.get_model):
            return trainer.get_model()

        return None

    def _extract_scaler(self, trainer: Any) -> Any | None:
        """
        Extract scaler from trainer instance.

        Uses TrainerProtocol when available, falls back to duck-typing.

        Args:
            trainer: Trainer instance

        Returns:
            Scaler instance or None
        """
        proto = _get_trainer_protocol()
        if proto is not None and isinstance(trainer, proto):
            return trainer.scaler

        for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:
            scaler = getattr(trainer, attr, None)
            if scaler is not None:
                return scaler
        return None

    def _extract_feature_columns(
        self,
        trainer: Any,
        n_features: int,
    ) -> list[str]:
        """
        Extract feature column names from trainer.

        Uses TrainerProtocol when available, falls back to duck-typing.

        Args:
            trainer: Trainer instance
            n_features: Number of features (fallback for generic names)

        Returns:
            List of feature column names
        """
        proto = _get_trainer_protocol()
        if proto is not None and isinstance(trainer, proto):
            cols = trainer.feature_columns
            if cols is not None and len(cols) > 0:
                return list(cols)

        # Legacy duck-typing fallback
        for attr in ["feature_columns", "_feature_columns", "feature_names", "_feature_names"]:
            columns = getattr(trainer, attr, None)
            if columns is not None and len(columns) > 0:
                return list(columns)

        # Try to get from scaler if available
        scaler = self._extract_scaler(trainer)
        if scaler is not None:
            columns = getattr(scaler, "feature_names_in_", None)
            if columns is not None:
                return list(columns)

        # Fallback to generic names
        logger.warning("No feature columns found, using generic names")
        return [f"f{i}" for i in range(n_features)]

    def _extract_calibrator(self, trainer: Any) -> Any | None:
        """
        Extract probability calibrator from trainer.

        Uses TrainerProtocol when available, falls back to duck-typing.

        Args:
            trainer: Trainer instance

        Returns:
            Calibrator instance or None
        """
        proto = _get_trainer_protocol()
        if proto is not None and isinstance(trainer, proto):
            return trainer.calibrator

        for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
            calibrator = getattr(trainer, attr, None)
            if calibrator is not None:
                return calibrator
        return None

    def _auto_generate_feature_spec(
        self,
        model_name: str,
        feature_columns: list[str],
        horizon: int,
        model_result: Any,
    ) -> Any | None:
        """
        Auto-generate a basic FeatureSpec from training result metadata.

        Best-effort: returns None if FeatureSpec cannot be created (e.g.,
        missing required fields). This ensures bundles still work without
        explicit FeatureSpec while providing one when possible.

        Args:
            model_name: Model name (e.g., "xgboost")
            feature_columns: Feature column names from the trainer
            horizon: Prediction horizon in bars
            model_result: The per-model training result

        Returns:
            FeatureSpec instance or None
        """
        try:
            from src.core.contracts.feature_spec import FeatureSpec

            # Extract hyperparameters from model_result if available
            hyperparameters: dict[str, Any] = {}
            if hasattr(model_result, "metrics") and isinstance(model_result.metrics, dict):
                hyperparameters = model_result.metrics.get("hyperparameters", {})

            spec = FeatureSpec(
                profit_threshold=(
                    self.config.profit_threshold
                    if hasattr(self.config, "profit_threshold")
                    else 0.015
                ),
                loss_threshold=(
                    self.config.loss_threshold if hasattr(self.config, "loss_threshold") else 0.010
                ),
                max_holding_bars=horizon,
                selected_features=list(feature_columns),
                model_name=model_name,
                hyperparameters=hyperparameters,
            )
            logger.debug(f"Auto-generated FeatureSpec for {model_name}: {spec.n_features} features")
            return spec
        except Exception as e:
            logger.debug(f"Could not auto-generate FeatureSpec for {model_name}: {e}")
            return None

    def _calculate_total_size(self, paths: list[Path]) -> float:
        """
        Calculate total size of bundles in MB.

        Args:
            paths: List of bundle paths (directories)

        Returns:
            Total size in megabytes
        """
        total_bytes = 0
        for path in paths:
            if path.exists():
                if path.is_dir():
                    total_bytes += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                else:
                    total_bytes += path.stat().st_size
        return total_bytes / (1024 * 1024)

    def validate_bundles(self) -> dict[str, Any]:
        """
        Validate all bundles in the bundles directory.

        Returns:
            Dictionary with validation results for each bundle
        """
        from src.inference.bundle import ModelBundle

        results: dict[str, Any] = {
            "valid": True,
            "bundles": {},
            "issues": [],
        }

        # Find all bundle directories
        if not self.bundles_dir.exists():
            results["valid"] = False
            results["issues"].append("Bundles directory does not exist")
            return results

        for bundle_dir in self.bundles_dir.iterdir():
            if not bundle_dir.is_dir():
                continue
            if bundle_dir.name == "ensemble":
                # Handle ensemble bundle separately
                continue

            try:
                bundle = ModelBundle.load(bundle_dir)
                validation = bundle.validate()
                results["bundles"][bundle_dir.name] = validation

                if not validation["valid"]:
                    results["valid"] = False
                    for issue in validation.get("issues", []):
                        results["issues"].append(f"{bundle_dir.name}: {issue}")

            except Exception as e:
                results["valid"] = False
                results["bundles"][bundle_dir.name] = {
                    "valid": False,
                    "error": str(e),
                }
                results["issues"].append(f"{bundle_dir.name}: Failed to load - {e}")

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_bundles(
    config: PipelineConfig,
    training_result: TrainingRunResult,
    ensemble_result: EnsembleResult | None = None,
    feature_specs: dict[str, Any] | None = None,
) -> BundleBuildResult:
    """
    Convenience function to build all bundles.

    Creates a BundleBuilder and builds bundles from the provided results.

    Usage:
        from src.core import PipelineConfig
        from src.inference.builder import build_bundles

        result = build_bundles(config, training_result, ensemble_result)
        print(f"Created {result.n_bundles} bundles")

    Args:
        config: PipelineConfig instance
        training_result: PHASE_3 training result
        ensemble_result: Optional PHASE_4 ensemble result
        feature_specs: Optional dict mapping model_key -> FeatureSpec

    Returns:
        BundleBuildResult with all bundle paths
    """
    builder = BundleBuilder(config)
    return builder.build_all(training_result, ensemble_result, feature_specs=feature_specs)


def build_from_run(
    run_path: str | Path,
    training_result: TrainingRunResult,
    ensemble_result: EnsembleResult | None = None,
    feature_specs: dict[str, Any] | None = None,
) -> BundleBuildResult:
    """
    Build bundles from a completed training run.

    Loads config from the run directory and builds bundles.

    Args:
        run_path: Path to training run directory
        training_result: PHASE_3 training result
        ensemble_result: Optional PHASE_4 ensemble result
        feature_specs: Optional dict mapping model_key -> FeatureSpec

    Returns:
        BundleBuildResult with all bundle paths
    """
    builder = BundleBuilder.from_training_run(run_path)
    return builder.build_all(training_result, ensemble_result, feature_specs=feature_specs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BundleBuilder",
    "BundleBuildResult",
    "build_bundles",
    "build_from_run",
]
