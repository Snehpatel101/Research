# ExperimentConfig

Every field with its default and allowed values is tabulated on the
[Configuration](../configuration.md) page; this page documents the classes and
their methods.

::: src.config.experiment.ExperimentConfig
    options:
      members:
        - from_dict
        - from_yaml
        - to_dict
        - save_yaml
        - resolve_barrier_params
        - label_span_bars
        - resolve_cv_gaps
        - to_pipeline_config

## Sections

::: src.config.experiment.DataSection

::: src.config.experiment.TrainingSection

::: src.config.experiment.RegimeSettings

::: src.config.experiment.MetaLabelingSettings

::: src.config.experiment.EvaluationSection

::: src.config.experiment.BundlingSection

## Sub-configs

::: src.config.data.LabelingConfig
    options:
      members: false

::: src.config.data.SequenceConfig
    options:
      members: false

::: src.config.data.MTFConfig
    options:
      members: false

::: src.config.data.SplitConfig
    options:
      members: false

::: src.config.data.FeatureConfig
    options:
      members: false

::: src.config.cv.WalkForwardConfig
    options:
      members: false

::: src.config.training.OptunaConfig
    options:
      members: false

::: src.config.training.CalibrationConfig
    options:
      members: false
