import pandas as pd

from src.manifest import ArtifactManifest
from src.pipeline_config import PipelineConfig
from src.pipeline.stages.datasets import run_build_datasets


def _write_scaled_splits(base_dir):
    scaled_dir = base_dir / "data" / "splits" / "scaled"
    scaled_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="min"),
            "symbol": ["MES"] * 5,
            "open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.05, 1.15, 1.25, 1.35, 1.45],
            "volume": [100, 110, 120, 130, 140],
            "rsi_14": [50, 52, 54, 56, 58],
            "macd_hist": [0.1, 0.2, 0.1, 0.0, -0.1],
            "label_h5": [1, 0, -1, 1, 0],
            "sample_weight_h5": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    df.to_parquet(scaled_dir / "train_scaled.parquet", index=False)
    df.to_parquet(scaled_dir / "val_scaled.parquet", index=False)
    df.to_parquet(scaled_dir / "test_scaled.parquet", index=False)


def test_run_build_datasets(tmp_path):
    config = PipelineConfig(
        project_root=tmp_path,
        run_id="test_run",
        symbols=["MES"],
        label_horizons=[5],
        feature_set="core_full",
    )
    config.create_directories()

    _write_scaled_splits(tmp_path)

    manifest = ArtifactManifest(config.run_id, config.project_root)
    result = run_build_datasets(config, manifest)

    assert result.status.name == "COMPLETED"
    dataset_manifest = config.run_artifacts_dir / "dataset_manifest.json"
    assert dataset_manifest.exists()

    dataset_path = config.splits_dir / "datasets" / "core_full" / "h5" / "train.parquet"
    assert dataset_path.exists()
