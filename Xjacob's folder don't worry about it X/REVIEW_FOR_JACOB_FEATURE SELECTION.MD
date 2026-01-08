To apply dynamic feature selection safely in this pipeline, I’d make it recomputed at retrain-time (walk-forward), per model family + horizon, and then frozen for that trained artifact (not changed per-row at inference).

Start with a candidate pool per model using feature sets (feature_sets.py) so each family sees the right “universe” (e.g., boosting_optimal vs neural_optimal vs transformer_raw).
For tabular/classical models, run walk-forward selection on X_train only using PurgedKFold + purge/embargo + label_end_times (leakage control) and keep stable features across folds (your code already does this via FeatureSelectionManager → WalkForwardFeatureSelector in manager.py and feature_selector.py).
Make it “dynamic” by repeating that selection separately per (symbol, horizon, model_name) each training run; persist the chosen feature list with the model (PersistedFeatureSelection via result.py).
Add redundancy control: clustered/correlation filtering (supported in feature_selector.py and OHLCV selectors like ohlcv_selector.py) so you don’t keep 10 near-duplicates.
If you want regime-aware dynamics, use regime-conditional importance (supported in ohlcv_selector.py) and either (a) select the union of top-k per regime or (b) keep a small fixed core + a small regime “addon” set.
For sequence/transformer models, don’t do hard external feature selection unless you keep input dims fixed; use the curated feature sets + model-internal gating/attention (your defaults disable external FS for neural in config.py).
If you tell me which “dynamic” you mean (per retrain window vs per-market-regime vs per-sample), I can suggest the exact protocol + knobs (n_features, min_feature_frequency, clustered importance) that fit your goal.

Yes? But The question is I"m trying to shove too many features? In so how do I found optimial features?


Yes—depending on your sample size and model family, you can absolutely be “shoving too many features”, especially if you’re feeding 150+ engineered features into tabular models on <100k samples (overfitting + unstable importance).

How to find “optimal” features in this pipeline (leakage-safe):

Use the built-in feature count heuristics first: feature_sets.py already encodes sample→feature guidance (e.g., 50–100 features is often the sweet spot unless you have hundreds of thousands of rows).
Start with a smaller candidate universe (not selection yet): pick feature_set=boosting_optimal or core_min rather than mtf_plus/core_full so you’re not optimizing inside a huge noisy space.
Treat “#features” as a hyperparameter: for tabular/classical models run walk-forward selection with PurgedKFold and sweep n_features (e.g., 20, 40, 60, 80, 100) and choose the best OOF metric; your code path is manager.py → feature_selector.py.
Require stability, not just performance: keep features that appear consistently across folds (controlled by min_feature_frequency in config.py), otherwise you’re selecting noise.
Prune redundancy: enable clustered/correlation-aware selection so you don’t keep 10 correlated variants of RSI/returns; the selectors support this (use_clustered_importance / correlation logic in feature_selector.py and ohlcv_selector.py).
Lock the result per (symbol, horizon, model) and only judge once on the test set at the very end—don’t tune features using test performance.
If you tell me: (1) approx #rows after your split, (2) which model you care about first (CatBoost/XGBoost vs TCN/PatchTST), and (3) whether you’re using MTF features, I’ll give you a concrete starting range for n_features and which feature_set to begin with.