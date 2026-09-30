# Deploy and bundles

How these fit together is described in [Deploy and serve](../deploy-and-serve.md).

## Loading a deployed run

::: src.inference.deploy.load_deploy_artifact

::: src.inference.deploy.load_bundle

::: src.inference.deploy.select_deploy_artifact

::: src.inference.deploy.validate_deploy_artifact

::: src.inference.deploy.describe_bundle

::: src.inference.deploy.DeployManifest
    options:
      members: [save, load]

## Bundle kinds

::: src.inference.bundle.ModelBundle
    options:
      members:
        - load
        - save
        - predict_from_raw
        - raw_to_input
        - predict
        - preprocess
        - model_input
        - validate

::: src.inference.ensemble_bundle.EnsembleBundle
    options:
      members:
        - load
        - save
        - predict_from_raw
        - predict
        - predict_proba
        - validate
        - summary

::: src.inference.regime_bundle.RegimeBundle
    options:
      members:
        - load
        - save
        - predict_from_raw
        - detect_regimes
        - predict

::: src.inference.meta_labeling_bundle.MetaLabelingBundle
    options:
      members:
        - load
        - save
        - predict_from_raw
        - predict_meta
        - meta_probability
        - trade_mask
        - predict

::: src.inference.meta_labeling_bundle.MetaLabelingPrediction

## Prediction result

::: src.core.interfaces.PredictionResult
    options:
      members: false
