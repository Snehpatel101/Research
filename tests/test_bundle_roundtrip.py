"""
Behavioral tests for model bundle save -> load -> predict parity.

Covers the canonical production bundle layer (``src.inference.bundle.ModelBundle``),
which is exactly what ``BundleBuilder.build_from_training_result()`` drives in
production (from_training -> save -> load -> predict).

Verifies:
1. Save produces the documented directory layout (manifest, metadata, features,
   scaler, model dir) with valid checksums.
2. Metadata records the model name / family / horizon / features correctly,
   both in memory and after a disk round trip.
3. Loaded bundle predictions are exactly equal to the original bundle's
   predictions (class predictions, probabilities, confidence).
4. DataFrame inputs are reordered by feature name (column order independent).
5. Missing / mis-shaped inputs raise clear errors.
6. Overwrite semantics, tarball package/extract round trip, and validate().

Uses a tiny XGBoost model (2D tabular, CPU-only) on synthetic data:
500 rows x 10 features. No neural models, no real data files.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.inference.bundle import (
    BUNDLE_VERSION,
    BundleMetadata,
    ModelBundle,
)
from src.models.base import PredictionResult
from src.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROWS = 500
N_TRAIN = 400
N_FEATURES = 10
HORIZON = 5
SYMBOL = "MES"
FEATURE_COLUMNS = [f"feat_{i:02d}" for i in range(N_FEATURES)]

MODEL_CONFIG = {
    "n_estimators": 30,
    "max_depth": 3,
    "learning_rate": 0.1,
    "early_stopping_rounds": 10,
    "use_gpu": False,
    "n_jobs": 1,
    "random_state": 42,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Module-scoped fixtures: train once, reuse across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifacts() -> dict:
    """Train a tiny xgboost on synthetic tabular data and build a bundle."""
    rng = np.random.RandomState(42)
    X = rng.randn(N_ROWS, N_FEATURES).astype(np.float32)

    # Labels weakly dependent on feature 0 so all 3 classes are learnable
    signal = X[:, 0] + 0.5 * rng.randn(N_ROWS)
    y = np.where(signal > 0.4, 1, np.where(signal < -0.4, -1, 0)).astype(np.int64)

    # Ensure all 3 classes present in both splits (xgboost needs num_class=3)
    assert set(np.unique(y[:N_TRAIN])) == {-1, 0, 1}
    assert set(np.unique(y[N_TRAIN:])) == {-1, 0, 1}

    scaler = RobustScaler().fit(X[:N_TRAIN])
    X_scaled = scaler.transform(X).astype(np.float32)

    model = ModelRegistry.create("xgboost", config=MODEL_CONFIG)
    model.fit(X_scaled[:N_TRAIN], y[:N_TRAIN], X_scaled[N_TRAIN:], y[N_TRAIN:])

    bundle = ModelBundle.from_training(
        model=model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
        horizon=HORIZON,
        symbol=SYMBOL,
        training_metrics={"val_f1": 0.5},
        model_name="xgboost",
    )

    # Unscaled "new" data as a DataFrame — bundle.predict applies the scaler
    X_new = pd.DataFrame(X[N_TRAIN:].astype(np.float64), columns=FEATURE_COLUMNS)

    return {"bundle": bundle, "scaler": scaler, "X_new": X_new}


@pytest.fixture(scope="module")
def bundle_dir(artifacts: dict, tmp_path_factory: pytest.TempPathFactory):
    """Save the bundle once for all read-only tests."""
    path = tmp_path_factory.mktemp("bundles") / "xgboost_h5"
    saved = artifacts["bundle"].save(path)
    return saved


@pytest.fixture(scope="module")
def loaded_bundle(bundle_dir) -> ModelBundle:
    """Load the bundle via the canonical load path."""
    return ModelBundle.load(bundle_dir)


# ---------------------------------------------------------------------------
# Save: directory layout and manifest
# ---------------------------------------------------------------------------


class TestBundleSave:
    def test_save_creates_expected_files(self, bundle_dir) -> None:
        assert (bundle_dir / "manifest.json").is_file()
        assert (bundle_dir / "metadata.json").is_file()
        assert (bundle_dir / "features.json").is_file()
        assert (bundle_dir / "scaler.pkl").is_file()
        assert (bundle_dir / "model").is_dir()
        # No calibrator was provided
        assert not (bundle_dir / "calibrator.pkl").exists()

    def test_manifest_lists_files_with_valid_checksums(self, bundle_dir) -> None:
        with open(bundle_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["version"] == BUNDLE_VERSION
        assert "metadata.json" in manifest["files"]
        assert "features.json" in manifest["files"]
        assert "scaler.pkl" in manifest["files"]
        assert "model" in manifest["files"]

        # Recompute MD5 checksums for each checksummed file
        for name, expected in manifest["checksums"].items():
            actual = hashlib.md5((bundle_dir / name).read_bytes()).hexdigest()
            assert actual == expected, f"Checksum mismatch for {name}"

    def test_features_json_preserves_column_order(self, bundle_dir) -> None:
        with open(bundle_dir / "features.json") as f:
            saved = json.load(f)["columns"]
        assert saved == FEATURE_COLUMNS

    def test_save_without_overwrite_raises(self, artifacts: dict, tmp_path) -> None:
        path = tmp_path / "dup_bundle"
        artifacts["bundle"].save(path)
        with pytest.raises(FileExistsError, match="already exists"):
            artifacts["bundle"].save(path)

    def test_save_with_overwrite_replaces_bundle(self, artifacts: dict, tmp_path) -> None:
        path = tmp_path / "ow_bundle"
        artifacts["bundle"].save(path)
        saved = artifacts["bundle"].save(path, overwrite=True)
        reloaded = ModelBundle.load(saved)
        assert reloaded.metadata.model_name == "xgboost"


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


class TestBundleMetadata:
    def test_metadata_records_model_name(self, artifacts: dict) -> None:
        meta = artifacts["bundle"].metadata
        assert meta.model_name == "xgboost"
        assert meta.model_family == "boosting"

    def test_metadata_fields(self, artifacts: dict) -> None:
        meta = artifacts["bundle"].metadata
        assert meta.version == BUNDLE_VERSION
        assert meta.horizon == HORIZON
        assert meta.n_features == N_FEATURES
        assert meta.symbol == SYMBOL
        assert meta.requires_sequences is False
        assert meta.requires_4d is False
        assert meta.has_calibrator is False
        assert meta.feature_names == FEATURE_COLUMNS
        assert meta.training_metrics == {"val_f1": 0.5}
        expected_hash = hashlib.md5(",".join(FEATURE_COLUMNS).encode()).hexdigest()[:12]
        assert meta.feature_hash == expected_hash

    def test_metadata_json_on_disk_has_model_name(self, bundle_dir) -> None:
        with open(bundle_dir / "metadata.json") as f:
            raw = json.load(f)
        assert raw["model_name"] == "xgboost"
        assert raw["model_family"] == "boosting"

    def test_loaded_metadata_matches_original(
        self, artifacts: dict, loaded_bundle: ModelBundle
    ) -> None:
        original = artifacts["bundle"].metadata
        loaded = loaded_bundle.metadata
        assert isinstance(loaded, BundleMetadata)
        assert loaded.to_dict() == original.to_dict()

    def test_loaded_feature_columns_match(
        self, artifacts: dict, loaded_bundle: ModelBundle
    ) -> None:
        assert loaded_bundle.feature_columns == artifacts["bundle"].feature_columns

    def test_loaded_scaler_matches_original(
        self, artifacts: dict, loaded_bundle: ModelBundle
    ) -> None:
        original = artifacts["scaler"]
        loaded = loaded_bundle.scaler
        assert loaded is not None
        np.testing.assert_array_equal(loaded.center_, original.center_)
        np.testing.assert_array_equal(loaded.scale_, original.scale_)
        assert loaded.n_features_in_ == N_FEATURES


# ---------------------------------------------------------------------------
# Predict parity: loaded predictions == original predictions exactly
# ---------------------------------------------------------------------------


class TestPredictParity:
    def test_loaded_predictions_exactly_match_original(
        self, artifacts: dict, loaded_bundle: ModelBundle
    ) -> None:
        X_new = artifacts["X_new"]

        result_orig = artifacts["bundle"].predict(X_new)
        result_loaded = loaded_bundle.predict(X_new)

        assert isinstance(result_orig, PredictionResult)
        assert isinstance(result_loaded, PredictionResult)

        np.testing.assert_array_equal(
            result_loaded.class_predictions, result_orig.class_predictions
        )
        np.testing.assert_array_equal(
            result_loaded.class_probabilities, result_orig.class_probabilities
        )
        np.testing.assert_array_equal(result_loaded.confidence, result_orig.confidence)

    def test_prediction_output_shapes_and_ranges(
        self, loaded_bundle: ModelBundle, artifacts: dict
    ) -> None:
        result = loaded_bundle.predict(artifacts["X_new"])
        n = len(artifacts["X_new"])

        assert result.class_predictions.shape == (n,)
        assert result.class_probabilities.shape == (n, 3)
        assert result.confidence.shape == (n,)
        assert set(np.unique(result.class_predictions)) <= {-1, 0, 1}
        np.testing.assert_allclose(result.class_probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(result.confidence > 0.0)
        assert np.all(result.confidence <= 1.0)

    def test_predict_is_column_order_independent(
        self, artifacts: dict, loaded_bundle: ModelBundle
    ) -> None:
        """DataFrame columns are reordered by name before prediction."""
        X_new = artifacts["X_new"]
        shuffled = X_new[list(reversed(FEATURE_COLUMNS))]

        result_ordered = loaded_bundle.predict(X_new)
        result_shuffled = loaded_bundle.predict(shuffled)

        np.testing.assert_array_equal(
            result_shuffled.class_predictions, result_ordered.class_predictions
        )
        np.testing.assert_array_equal(
            result_shuffled.class_probabilities, result_ordered.class_probabilities
        )

    def test_predict_missing_feature_raises(
        self, artifacts: dict, loaded_bundle: ModelBundle
    ) -> None:
        incomplete = artifacts["X_new"].drop(columns=[FEATURE_COLUMNS[3]])
        with pytest.raises(ValueError, match="Missing features"):
            loaded_bundle.predict(incomplete)

    def test_predict_wrong_feature_count_raises(self, loaded_bundle: ModelBundle) -> None:
        bad = np.zeros((5, N_FEATURES + 2), dtype=np.float32)
        with pytest.raises(ValueError, match="features"):
            loaded_bundle.predict(bad)


# ---------------------------------------------------------------------------
# Tarball deployment round trip
# ---------------------------------------------------------------------------


class TestPackageExtract:
    def test_package_extract_load_predict_parity(
        self, artifacts: dict, bundle_dir, tmp_path
    ) -> None:
        bundle = artifacts["bundle"]
        tarball = bundle.package_bundle(bundle_dir, output_path=tmp_path / "deploy.tar.gz")
        assert tarball.is_file()

        extract_dir = tmp_path / "extracted"
        extracted = ModelBundle.extract_bundle(tarball, extract_dir=extract_dir)
        reloaded = ModelBundle.load(extracted)

        assert reloaded.metadata.model_name == "xgboost"

        result_orig = bundle.predict(artifacts["X_new"])
        result_deploy = reloaded.predict(artifacts["X_new"])
        np.testing.assert_array_equal(
            result_deploy.class_predictions, result_orig.class_predictions
        )
        np.testing.assert_array_equal(
            result_deploy.class_probabilities, result_orig.class_probabilities
        )


# ---------------------------------------------------------------------------
# Validation and error paths
# ---------------------------------------------------------------------------


class TestValidationAndErrors:
    def test_loaded_bundle_validates_clean(self, loaded_bundle: ModelBundle) -> None:
        report = loaded_bundle.validate()
        assert report["valid"] is True
        assert report["issues"] == []
        assert report["metadata"]["model_name"] == "xgboost"

    def test_load_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Bundle not found"):
            ModelBundle.load(tmp_path / "does_not_exist")

    def test_load_dir_without_manifest_raises(self, tmp_path) -> None:
        empty = tmp_path / "not_a_bundle"
        empty.mkdir()
        with pytest.raises(ValueError, match="manifest"):
            ModelBundle.load(empty)

    def test_scaler_pickle_is_plain_sklearn_object(self, bundle_dir) -> None:
        """Scaler is stored as a plain pickle loadable by the safe loader."""
        with open(bundle_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)  # noqa: S301 - test-authored file
        assert isinstance(scaler, RobustScaler)
