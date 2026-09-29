# UniversalInferencePipeline

Serves several bundles (any mix of 2D, 3D and 4D models) plus an optional
ensemble behind one interface. For a single deployed artifact,
[`load_deploy_artifact`](inference.md) is simpler.

::: src.inference.universal_pipeline.UniversalInferencePipeline
    options:
      members:
        - from_bundle
        - from_bundles
        - from_experiment
        - from_training_result
        - predict_from_raw
        - predict
        - predict_all
        - predict_ensemble
        - predict_batch
        - predict_with_uncertainty
        - get_model_info
        - validate
        - summary
        - n_models
        - model_names
        - has_ensemble

::: src.inference.universal_pipeline.UniversalPredictionResult
