 - [High] PipelineState lacks status and error_message, but MLPipeline reads/writes them; resuming fails since state isn't persisted. refs src/
    ml_pipeline/unified.py:173, src/ml_pipeline/unified.py:186, src/ml_pipeline/unified.py:394, src/ml_pipeline/state.py:225
  - [High] Training dispatchers expect model_paths and metrics from trainer outputs, but inconsistencies across training modes result in empty
    evaluation inputs. refs src/ml_pipeline/unified.py:529, src/ml_pipeline/unified.py:550, src/training/orchestrator.py:77, src/training/
    orchestrator.py:155, src/training/modes/walk_forward.py:273, src/training/modes/regime_aware.py:459, src/training/modes/meta_labeling.py:403
  - [High] Default data_dir is "data" but training looks for data_dir/timeframe, while Phase 1 outputs to data/splits/scaled/<tf>; this mismatch
    causes FileNotFoundError in training. refs src/ml_pipeline/config.py:230, src/ml_pipeline/unified.py:506, src/training/orchestrator.py:100
  - [High] Default evaluation is ["cv"], but CVEvaluator.run() raises NotImplementedError, causing pipeline failures unless evaluation is
    disabled. refs src/ml_pipeline/config.py:183, src/evaluation/cv_evaluator.py:36
  - [Low] Feature metadata extraction reads parquet with empty columns=[], causing zero features found in features_info. refs src/ml_pipeline/
    unified.py:751 persisted. refs src/ml_pipeline/unified.py:173, src/ml_pipeline/unified.py:186, src/ml_pipeline/unified.py:394, src/ml_pipeline/state.py:225
  - [High] Training dispatchers expect model_paths/metrics from trainers, but TrainingOrchestrator and the mode trainers return different
    shapes, so training results are empty and evaluation gets no models. refs src/ml_pipeline/unified.py:529, src/ml_pipeline/unified.py:550,
    src/ml_pipeline/unified.py:571, src/ml_pipeline/unified.py:592, src/training/orchestrator.py:77, src/training/orchestrator.py:155, src/
    training/modes/walk_forward.py:273, src/training/modes/regime_aware.py:459, src/training/modes/meta_labeling.py:403
  - [High] Default data_dir is data, but training loads data_dir / timeframe, while Phase 1 outputs are under data/splits/scaled/<tf>; with
    defaults training will raise FileNotFoundError. refs src/ml_pipeline/config.py:230, src/ml_pipeline/unified.py:506, src/training/
    orchestrator.py:100
  - [High] Evaluation is enabled by default (evaluation_methods=["cv"]) but CVEvaluator.run() always raises NotImplementedError, so full runs
    fail unless evaluation is disabled. refs src/ml_pipeline/config.py:183, src/evaluation/cv_evaluator.py:36
  - [Low] Feature metadata extraction reads parquet with columns=[], which yields no columns; n_features/feature_names are always empty. ref
    src/ml_pipeline/unified.py:751