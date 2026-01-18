Build ONE cohesive ML factory from scratch. NO backward compatibility - clean slate approach.

     Goal: Single unified pipeline where data flows seamlessly from raw OHLCV to deployed inference bundle.

     ---
     IMPLEMENTATION STATUS: COMPLETE ✓

     THE SINGLE ENTRY POINT:
     ```python
     from src import MLFactory, PipelineConfig

     config = PipelineConfig(
         symbol="MES",
         data_path="./data/mes.parquet",
         output_dir="./experiments",
         models=["xgboost", "lightgbm", "lstm"],
         training_mode="standard",  # or "regime_aware", "meta_labeling"
         build_ensemble=True,
     )

     factory = MLFactory(config)
     result = factory.run(df)
     bundle = result.get_inference_bundle()
     ```

     KEY FILES (Single Source of Truth):
     - src/factory.py: MLFactory - THE single entry point
     - src/core/config.py: PipelineConfig - centralized configuration
     - src/core/interfaces.py: All abstract contracts
     - src/core/types.py: All enums and type definitions
     - src/core/constants.py: All constants

     RESOLVED ISSUES:
     ✓ Circular import (cross_validation → oof_core → models.ensemble → heterogeneous_stacking → oof_core)
       - Fixed using TYPE_CHECKING pattern in heterogeneous_stacking.py
       - OOFPredictionProtocol in src/core/interfaces.py breaks the cycle
     ✓ MTF feature computation - integrated in src/features/compute/mtf.py
     ✓ Regime-aware training - implemented in src/training/regime_trainer.py
     ✓ Meta-labeling training - implemented in src/training/unified_orchestrator.py

     ---                             
                                                                                                                                        
     ---                                                                                                                                
     COMPLETE FEATURE INVENTORY (Everything to Preserve)                                                                                
                                                                                                                                        
     22-23 Models (4 Families)                                                                                                          
                                                                                                                                        
     Boosting (2-3):                                                                                                                    
     - xgboost - XGBoost classifier                                                                                                     
     - lightgbm - LightGBM classifier                                                                                                   
     - catboost - CatBoost classifier (optional if installed)                                                                           
                                                                                                                                        
     Classical (3):                                                                                                                     
     - random_forest - Random Forest classifier                                                                                         
     - logistic - Logistic Regression                                                                                                   
     - svm - Support Vector Machine                                                                                                     
                                                                                                                                        
     Neural (10):                                                                                                                       
     - lstm - Long Short-Term Memory                                                                                                    
     - gru - Gated Recurrent Unit                                                                                                       
     - tcn - Temporal Convolutional Network                                                                                             
     - transformer - Self-attention Transformer                                                                                         
     - patchtst - Patched Time Series Transformer                                                                                       
     - itransformer - Inverted Transformer                                                                                              
     - tft - Temporal Fusion Transformer                                                                                                
     - nbeats - Neural Basis Expansion                                                                                                  
     - inceptiontime - Inception CNN                                                                                                    
     - resnet1d - 1D Residual Network                                                                                                   
                                                                                                                                        
     Ensemble + Meta-learners (7):                                                                                                      
     - voting - Soft/hard voting ensemble                                                                                               
     - stacking - OOF stacking with meta-learner                                                                                        
     - blending - Holdout set blending                                                                                                  
     - ridge_meta - L2-regularized meta-learner                                                                                         
     - mlp_meta - MLP meta-learner                                                                                                      
     - xgboost_meta - XGBoost meta-learner                                                                                              
     - calibrated_meta - Calibrated meta-learner                                                                                        
                                                                                                                                        
     162 Feature Indicators (12 Families) - See PHASE_1_UNIFIED_FEATURES.md for authoritative counts

     Momentum (7 functions, 23 features):                                                                                              
     - RSI (14-period + overbought/oversold flags)                                                                                      
     - MACD (12/26/9 + histogram + crossover signals)                                                                                   
     - Stochastic (%K, %D, smoothed)                                                                                                    
     - Williams %R                                                                                                                      
     - ROC (rate of change)                                                                                                             
     - CCI (commodity channel index)                                                                                                    
     - MFI (money flow index)                                                                                                           
                                                                                                                                        
     Moving Averages (4 functions, 16 features):                                                                                       
     - SMA (5, 10, 20, 50, 100, 200 periods)                                                                                            
     - EMA (5, 10, 20, 50, 100, 200 periods)                                                                                            
     - SMA crossovers (fast/slow signals)                                                                                               
     - Price-MA ratios                                                                                                                  
                                                                                                                                        
     Volatility (10 functions, ~25 features):                                                                                           
     - ATR (7, 14, 21 periods + normalized)                                                                                             
     - Bollinger Bands (upper/middle/lower + %B + width)                                                                                
     - Keltner Channels (upper/middle/lower + width)                                                                                    
     - Historical Volatility (20, 60 period)                                                                                            
     - Parkinson Volatility (high/low based)                                                                                            
     - Garman-Klass Volatility (OHLC based)                                                                                             
     - Rogers-Satchell Volatility                                                                                                       
     - Yang-Zhang Volatility                                                                                                            
     - Higher Moments (skewness, kurtosis)                                                                                              
     - GARCH Features (conditional volatility)                                                                                          
                                                                                                                                        
     Volume (6 functions, ~15 features):                                                                                                
     - OBV (On-Balance Volume)                                                                                                          
     - VWAP (session-based)                                                                                                             
     - TWAP                                                                                                                             
     - Dollar Volume                                                                                                                    
     - Volume MA (20-period)                                                                                                            
     - Volume Ratio                                                                                                                     
                                                                                                                                        
     Trend (2 functions, ~6 features):                                                                                                  
     - ADX (14-period + +DI/-DI + trend strength flag)                                                                                  
     - Supertrend (upper/lower bands + direction)                                                                                       
                                                                                                                                        
     Price Features (4 functions, 12 features):                                                                                         
     - Returns (simple + log)                                                                                                           
     - Price Ratios (high/low, close/open)                                                                                              
     - Autocorrelation (multi-lag)                                                                                                      
     - CLV (Close Location Value)                                                                                                       
                                                                                                                                        
     Microstructure (9 functions, ~15 features):                                                                                        
     - Amihud Illiquidity                                                                                                               
     - Roll Spread                                                                                                                      
     - Kyle Lambda                                                                                                                      
     - Corwin-Schultz Spread                                                                                                            
     - Relative Spread                                                                                                                  
     - Volume Imbalance                                                                                                                 
     - Trade Intensity                                                                                                                  
     - Price Efficiency                                                                                                                 
     - Realized Volatility Ratio                                                                                                        
                                                                                                                                        
     Entropy (5 functions, 12 features):                                                                                                
     - Shannon Entropy                                                                                                                  
     - Lempel-Ziv Complexity                                                                                                            
     - Approximate Entropy                                                                                                              
     - Sample Entropy                                                                                                                   
     - Hurst Exponent                                                                                                                   
                                                                                                                                        
     Wavelets (5 functions, ~15 features):                                                                                              
     - DWT Coefficients (db4, db8, sym5, coif3, haar)                                                                                   
     - Wavelet Approximation                                                                                                            
     - Wavelet Details (3 levels)                                                                                                       
     - Wavelet Energy                                                                                                                   
     - Wavelet Entropy                                                                                                                  
                                                                                                                                        
     Temporal (4 functions, 9 features):                                                                                               
     - Hour of Day (sin/cos encoded)                                                                                                    
     - Day of Week (sin/cos encoded)                                                                                                    
     - Time Since Open                                                                                                                  
     - Time to Close                                                                                                                    
     - Session Progress                                                                                                                 
     - End-of-Session flag                                                                                                              
                                                                                                                                        
     Regime (4 functions, ~9 features):                                                                                                 
     - Volatility Regime (low/normal/high - ATR percentile)                                                                             
     - Trend Regime (downtrend/sideways/uptrend - ADX based)                                                                            
     - Composite Regime (9 combinations)                                                                                                
     - Regime indicators                                                                                                                
                                                                                                                                        
     MTF Features (~30+ features per higher TF):                                                                                        
     - 9 Intraday Timeframes: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h                                                                  
     - All indicators replicated per TF with shift(1) anti-lookahead                                                                    
                                                                                                                                        
     3 Training Modes                                                                                                                   
                                                                                                                                        
     Walk-Forward Validation:                                                                                                           
     - Expanding or sliding windows                                                                                                     
     - Proper purge/embargo handling                                                                                                    
     - Aggregate metrics across windows                                                                                                 
     - Classes: WalkForwardTrainerConfig, WalkForwardTrainer                                                                            
                                                                                                                                        
     Regime-Aware Training:                                                                                                             
     - Volatility regimes: low_vol, high_vol                                                                                            
     - Trend regimes: trending, mean_reverting                                                                                          
     - Composite: 4 combinations                                                                                                        
     - Train separate models OR single model with regime features                                                                       
     - Classes: RegimeAwareConfig, RegimeAwareTrainer                                                                                   
                                                                                                                                        
     Meta-Labeling (Lopez de Prado):                                                                                                    
     - Primary model: direction prediction                                                                                              
     - Meta-model: bet sizing based on confidence                                                                                       
     - Combined: direction × bet_size                                                                                                   
     - Classes: MetaLabelingConfig, MetaLabelingTrainer                                                                                 
                                                                                                                                        
     4 Cross-Validation Methods                                                                                                         
                                                                                                                                        
     PurgedKFold:                                                                                                                       
     - Configurable purge_bars (default 60)                                                                                             
     - Configurable embargo_bars (default 1440)                                                                                         
     - Time-series aware (no shuffling)                                                                                                 
     - Classes: PurgedKFoldConfig, PurgedKFold                                                                                          
                                                                                                                                        
     Walk-Forward CV:                                                                                                                   
     - Expanding/rolling windows                                                                                                        
     - Gap and embargo handling                                                                                                         
     - Classes: WalkForwardConfig, WalkForwardEvaluator                                                                                 
                                                                                                                                        
     CPCV (Combinatorially Purged CV):                                                                                                  
     - Multiple independent path combinations                                                                                           
     - Robust overfitting detection                                                                                                     
     - Classes: CPCVConfig, CombinatorialPurgedCV                                                                                       
                                                                                                                                        
     PBO (Probability of Backtest Overfitting):                                                                                         
     - Detect backtest selection bias                                                                                                   
     - DSR integration                                                                                                                  
     - Classes: PBOConfig, PBOResult                                                                                                    
                                                                                                                                        
     OOF Generation System                                                                                                              
                                                                                                                                        
     Tabular OOF:                                                                                                                       
     - CoreOOFGenerator - Core tabular OOF                                                                                              
     - OOFGenerator - Unified interface                                                                                                 
                                                                                                                                        
     Sequence OOF:                                                                                                                      
     - SequenceOOFGenerator - 3D sequence OOF                                                                                           
                                                                                                                                        
     Stacking:                                                                                                                          
     - StackingDatasetBuilder - Homogeneous stacking                                                                                    
     - HeterogeneousStackingBuilder - Mixed tabular+sequence                                                                            
     - OOFAlignmentValidator - Alignment validation                                                                                     
     - OOFCache - Intelligent caching                                                                                                   
                                                                                                                                        
     Inference System                                                                                                                   
                                                                                                                                        
     Model Bundling:                                                                                                                    
     - ModelBundle - Serializable container (V1.1.0)                                                                                    
     - BundleMetadata - Model metadata                                                                                                  
     - PreprocessingGraph - Feature lineage for raw inference                                                                           
                                                                                                                                        
     Inference Pipeline:                                                                                                                
     - InferencePipeline - Single/ensemble predictions                                                                                  
     - BatchPredictor - Chunked batch processing                                                                                        
     - ModelServer - FastAPI HTTP serving                                                                                               
                                                                                                                                        
     Validation System                                                                                                                  
                                                                                                                                        
     - LookaheadAuditor - Detect forward-looking features                                                                               
     - compute_deflated_sharpe() - Selection bias detection                                                                             
     - StatisticalTestResult - Model comparison tests                                                                                   
     - bootstrap_*() - CI estimation functions                                                                                          
     - comprehensive_leakage_check() - Leakage detection                                                                                
                                                                                                                                        
     State Management                                                                                                                   
                                                                                                                                        
     - PipelineState - Thread-safe state tracking                                                                                       
     - PhaseState enum: NOT_STARTED, IN_PROGRESS, COMPLETED, FAILED, SKIPPED                                                            
     - PhaseRegistry - Dependency graph with topological ordering                                                                       
     - StateValidator - Consistency validation                                                                                          
                                                                                                                                        
     Data Adapters                                                                                                                      
                                                                                                                                        
     - TabularAdapter - 2D (n_samples, n_features)                                                                                      
     - SequenceAdapter - 3D (n_samples, seq_len, n_features)                                                                            
     - MultiStreamAdapter - 4D (n_samples, n_timeframes, seq_len, n_features)                                                           
                                                                                                                                        
     6 Labeling Methods                                                                                                                 
                                                                                                                                        
     - Triple-Barrier (primary - ATR-based barriers)                                                                                    
     - Adaptive Triple-Barrier (regime-dependent barriers)                                                                              
     - Directional (return sign based)                                                                                                  
     - Threshold (percentage thresholds)                                                                                                
     - Regression (continuous return targets)                                                                                           
     - Meta-Labeling (confidence labels)                                                                                                
                                                                                                                                        
     Pipeline Stages (16 total) - See HIGH_LEVEL_DATA_FLOW.md for detailed diagrams

     1. Ingestion - Load raw OHLCV
     2. Cleaning - Resample, gap handling
     3. Sessions - Trading hours filtering
     4. MTF Upscaling - 9 timeframes from 1-min
     5. Features - 162 indicators
     6. Regime - Market regime detection
     7. OPTUNA: Label Optimization - Triple Barrier parameter optimization
     8. OPTUNA: Feature Selection - Binary include/exclude optimization
     9. OPTUNA: Feature Pruning - Importance-based feature removal
     10. Splits - Train/val/test (70/15/15)
     11. Scaling - Train-only robust scaling
     12. Adaptation - 2D/3D/4D tensor per model type
     13. OPTUNA: Hyperparameter Optimization - Model-specific hyperparameter tuning
     14. Training - PurgedKFold CV, OOF generation
     15. Stacking - OOF alignment, meta-learner
     16. Bundling - Model + Scaler + Graph -> Artifact

     ---
     DETAILED OPTUNA OPTIMIZATION STAGES (PHASE_1B)

     **Stage 7: OPTUNA Label Optimization (Triple Barrier)**

     Purpose: Find optimal triple-barrier labeling parameters that maximize risk-adjusted returns.

     Parameters to optimize:
     - upper_mult: Upper barrier multiplier (0.5 to 3.0 x ATR)
     - lower_mult: Lower barrier multiplier (0.5 to 3.0 x ATR)
     - horizon: Maximum holding period in bars (10 to 100 bars)
     - atr_period: ATR lookback period (7 to 28 bars)

     Optimization settings:
     - Trials: 100
     - Objective: Maximize Sharpe ratio with transaction cost penalty
     - Sampler: TPE (Tree-structured Parzen Estimator)
     - Pruner: MedianPruner with n_startup_trials=10

     Search space bounds (symbol-dependent):
     ```python
     LABEL_SEARCH_SPACE = {
         "MES": {
             "upper_mult": (1.0, 2.5),
             "lower_mult": (1.0, 2.5),
             "horizon": (20, 60),
             "atr_period": (10, 21),
         },
         "MNQ": {
             "upper_mult": (1.5, 3.0),
             "lower_mult": (1.5, 3.0),
             "horizon": (15, 50),
             "atr_period": (10, 21),
         },
         # Default fallback for other symbols
         "default": {
             "upper_mult": (0.5, 3.0),
             "lower_mult": (0.5, 3.0),
             "horizon": (10, 100),
             "atr_period": (7, 28),
         },
     }
     ```

     Code example:
     ```python
     import optuna
     from src.labels import TripleBarrierLabeler
     from src.validation import compute_sharpe_with_costs

     def label_objective(trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
         bounds = LABEL_SEARCH_SPACE.get(symbol, LABEL_SEARCH_SPACE["default"])

         upper_mult = trial.suggest_float("upper_mult", *bounds["upper_mult"])
         lower_mult = trial.suggest_float("lower_mult", *bounds["lower_mult"])
         horizon = trial.suggest_int("horizon", *bounds["horizon"])
         atr_period = trial.suggest_int("atr_period", *bounds["atr_period"])

         labeler = TripleBarrierLabeler(
             upper_mult=upper_mult,
             lower_mult=lower_mult,
             horizon=horizon,
             atr_period=atr_period,
         )
         labels = labeler.fit_transform(df)

         # Evaluate label quality via quick model + Sharpe
         sharpe = compute_sharpe_with_costs(labels, df["close"], cost_bps=2.0)
         return sharpe

     study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
     study.optimize(lambda t: label_objective(t, df, symbol), n_trials=100)
     best_label_params = study.best_params
     ```

     ---
     **Stage 8: OPTUNA Feature Selection Optimization**

     Purpose: Select optimal subset of features via binary include/exclude decisions.

     Optimization approach:
     - Binary decision for each feature (include=1, exclude=0)
     - 162 binary parameters (one per feature)
     - Minimum feature constraint: at least 10 features must be selected

     Optimization settings:
     - Trials: 100
     - Objective: Maximize model performance (F1-score or Sharpe ratio)
     - Sampler: TPE with multivariate=True for feature correlation
     - Pruner: SuccessiveHalvingPruner

     Features grouped by family for structured selection:
     ```python
     FEATURE_GROUPS = {
         "momentum": ["rsi_14", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", ...],
         "moving_averages": ["sma_5", "sma_10", "sma_20", "ema_5", "ema_10", ...],
         "volatility": ["atr_14", "bb_upper", "bb_lower", "bb_width", "keltner_upper", ...],
         "volume": ["obv", "vwap", "dollar_volume", "volume_ma_20", ...],
         "trend": ["adx_14", "plus_di", "minus_di", "supertrend", ...],
         "price": ["returns", "log_returns", "high_low_ratio", "close_open_ratio", ...],
         "microstructure": ["amihud", "roll_spread", "kyle_lambda", ...],
         "entropy": ["shannon_entropy", "lz_complexity", "approx_entropy", ...],
         "wavelets": ["dwt_approx", "dwt_detail_1", "dwt_detail_2", ...],
         "temporal": ["hour_sin", "hour_cos", "day_sin", "day_cos", ...],
         "regime": ["vol_regime", "trend_regime", "composite_regime", ...],
         "mtf": ["mtf_5m_rsi", "mtf_15m_macd", "mtf_1h_atr", ...],
     }
     ```

     Code example:
     ```python
     import optuna
     from src.features import FEATURE_REGISTRY
     from src.training import quick_evaluate_features

     def feature_selection_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
         selected_features = []

         for feature_name in FEATURE_REGISTRY.keys():
             include = trial.suggest_categorical(f"include_{feature_name}", [0, 1])
             if include:
                 selected_features.append(feature_name)

         # Enforce minimum feature count
         if len(selected_features) < 10:
             return float("-inf")

         X_selected = X[selected_features]
         score = quick_evaluate_features(X_selected, y, model="lightgbm", cv_folds=3)
         return score

     study = optuna.create_study(
         direction="maximize",
         sampler=optuna.samplers.TPESampler(multivariate=True),
     )
     study.optimize(lambda t: feature_selection_objective(t, X, y), n_trials=100)
     selected_features = [f for f in FEATURE_REGISTRY if study.best_params.get(f"include_{f}", 0)]
     ```

     ---
     **Stage 9: OPTUNA Feature Pruning Optimization**

     Purpose: Remove low-importance features based on model-derived importance scores.

     Optimization approach:
     - Train base model to get feature importances
     - Optimize importance threshold for feature removal
     - Iteratively prune features below threshold
     - Reduces feature count from ~162 to optimal subset (typically 40-80 features)

     Optimization settings:
     - Trials: 50
     - Objective: Maximize performance while minimizing feature count
     - Multi-objective: Pareto frontier of performance vs. complexity

     Pruning parameters:
     ```python
     PRUNING_SEARCH_SPACE = {
         "importance_threshold": (0.001, 0.05),  # Minimum importance to keep
         "cumulative_importance": (0.85, 0.99),   # Keep top N% cumulative importance
         "max_features": (30, 100),               # Hard cap on feature count
         "correlation_threshold": (0.85, 0.98),   # Remove correlated features
     }
     ```

     Code example:
     ```python
     import optuna
     import lightgbm as lgb
     from src.features import compute_feature_importance, remove_correlated

     def feature_pruning_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
         importance_threshold = trial.suggest_float("importance_threshold", 0.001, 0.05)
         cumulative_target = trial.suggest_float("cumulative_importance", 0.85, 0.99)
         correlation_threshold = trial.suggest_float("correlation_threshold", 0.85, 0.98)

         # Get feature importances from LightGBM
         model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
         model.fit(X, y)
         importances = pd.Series(model.feature_importances_, index=X.columns)
         importances = importances / importances.sum()  # Normalize

         # Prune by importance threshold
         keep_mask = importances >= importance_threshold

         # Prune by cumulative importance
         sorted_imp = importances.sort_values(ascending=False)
         cumsum = sorted_imp.cumsum()
         keep_by_cumulative = cumsum[cumsum <= cumulative_target].index.tolist()

         # Combine criteria
         features_to_keep = list(set(importances[keep_mask].index) & set(keep_by_cumulative))

         # Remove highly correlated features
         X_pruned = remove_correlated(X[features_to_keep], threshold=correlation_threshold)

         # Evaluate pruned feature set
         score = quick_evaluate_features(X_pruned, y, model="lightgbm", cv_folds=3)

         # Penalize for too many features (complexity penalty)
         complexity_penalty = len(X_pruned.columns) / 162 * 0.1
         return score - complexity_penalty

     study = optuna.create_study(direction="maximize")
     study.optimize(lambda t: feature_pruning_objective(t, X, y), n_trials=50)
     ```

     ---
     **Stage 13: OPTUNA Hyperparameter Optimization**

     Purpose: Tune model-specific hyperparameters for all 23 models.

     Optimization settings:
     - Trials: 100 per model
     - Total trials: 100 x 23 = 2,300 trials
     - Objective: Maximize validation F1-score (or Sharpe for trading)
     - CV: PurgedKFold with 5 folds
     - Sampler: TPE for most, CMA-ES for neural networks

     Search spaces by model family:

     **Boosting Models (XGBoost, LightGBM, CatBoost):**
     ```python
     BOOSTING_SEARCH_SPACE = {
         "n_estimators": (100, 1000),
         "max_depth": (3, 12),
         "learning_rate": (0.01, 0.3, "log"),
         "min_child_weight": (1, 10),
         "subsample": (0.6, 1.0),
         "colsample_bytree": (0.6, 1.0),
         "reg_alpha": (1e-8, 10.0, "log"),
         "reg_lambda": (1e-8, 10.0, "log"),
         "gamma": (0, 5),  # XGBoost only
         "num_leaves": (20, 150),  # LightGBM only
         "bagging_fraction": (0.6, 1.0),  # LightGBM only
     }
     ```

     Code example (XGBoost):
     ```python
     def xgboost_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
         params = {
             "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
             "max_depth": trial.suggest_int("max_depth", 3, 12),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
             "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
             "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
             "gamma": trial.suggest_float("gamma", 0, 5),
             "tree_method": "hist",
             "eval_metric": "logloss",
         }

         model = xgb.XGBClassifier(**params, use_label_encoder=False)
         scores = cross_val_score(model, X, y, cv=PurgedKFold(n_splits=5), scoring="f1_weighted")
         return scores.mean()
     ```

     **Neural Network Models (LSTM, GRU, TCN, Transformer, etc.):**
     ```python
     NEURAL_SEARCH_SPACE = {
         "hidden_size": (32, 256),
         "num_layers": (1, 4),
         "dropout": (0.1, 0.5),
         "learning_rate": (1e-5, 1e-2, "log"),
         "batch_size": [32, 64, 128, 256],
         "weight_decay": (1e-6, 1e-2, "log"),
         "attention_heads": (2, 8),  # Transformer only
         "kernel_size": (2, 7),  # TCN only
         "num_blocks": (1, 4),  # N-BEATS only
         "patch_size": (8, 32),  # PatchTST only
     }
     ```

     Code example (LSTM):
     ```python
     def lstm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
         params = {
             "hidden_size": trial.suggest_int("hidden_size", 32, 256),
             "num_layers": trial.suggest_int("num_layers", 1, 4),
             "dropout": trial.suggest_float("dropout", 0.1, 0.5),
             "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
             "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
             "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         }

         model = LSTMClassifier(**params)
         trainer = NeuralTrainer(model, early_stopping_patience=10)
         score = trainer.cross_validate(X, y, cv=PurgedKFold(n_splits=5))
         return score
     ```

     **Classical Models (Random Forest, Logistic Regression, SVM):**
     ```python
     CLASSICAL_SEARCH_SPACE = {
         # Random Forest
         "rf_n_estimators": (100, 500),
         "rf_max_depth": (5, 30),
         "rf_min_samples_split": (2, 20),
         "rf_min_samples_leaf": (1, 10),
         "rf_max_features": ["sqrt", "log2", None],

         # Logistic Regression
         "lr_C": (1e-4, 100, "log"),
         "lr_penalty": ["l1", "l2", "elasticnet"],
         "lr_solver": ["saga", "lbfgs"],
         "lr_l1_ratio": (0.0, 1.0),  # For elasticnet

         # SVM
         "svm_C": (1e-3, 100, "log"),
         "svm_kernel": ["rbf", "poly", "sigmoid"],
         "svm_gamma": ["scale", "auto"],
         "svm_degree": (2, 5),  # For poly kernel
     }
     ```

     Code example (Random Forest):
     ```python
     def rf_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
         params = {
             "n_estimators": trial.suggest_int("n_estimators", 100, 500),
             "max_depth": trial.suggest_int("max_depth", 5, 30),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
             "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
             "n_jobs": -1,
         }

         model = RandomForestClassifier(**params)
         scores = cross_val_score(model, X, y, cv=PurgedKFold(n_splits=5), scoring="f1_weighted")
         return scores.mean()
     ```

     **Advanced/Ensemble Meta-Learners:**
     ```python
     ADVANCED_SEARCH_SPACE = {
         # Ridge Meta-Learner
         "ridge_alpha": (1e-4, 100, "log"),

         # MLP Meta-Learner
         "mlp_hidden_layers": [(64,), (128,), (64, 32), (128, 64), (256, 128, 64)],
         "mlp_activation": ["relu", "tanh"],
         "mlp_alpha": (1e-5, 1e-1, "log"),

         # XGBoost Meta-Learner
         "xgb_meta_n_estimators": (50, 300),
         "xgb_meta_max_depth": (2, 6),
         "xgb_meta_learning_rate": (0.01, 0.2, "log"),
     }
     ```

     Code example (Calibrated Meta-Learner):
     ```python
     def calibrated_meta_objective(trial: optuna.Trial, oof_preds: np.ndarray, y: np.ndarray) -> float:
         base_model = trial.suggest_categorical("base_model", ["ridge", "mlp", "xgboost"])

         if base_model == "ridge":
             alpha = trial.suggest_float("ridge_alpha", 1e-4, 100, log=True)
             model = CalibratedClassifierCV(RidgeClassifier(alpha=alpha), cv=3)
         elif base_model == "mlp":
             hidden = trial.suggest_categorical("mlp_hidden", [(64,), (128,), (64, 32)])
             model = CalibratedClassifierCV(MLPClassifier(hidden_layer_sizes=hidden), cv=3)
         else:
             n_est = trial.suggest_int("xgb_n_estimators", 50, 300)
             model = CalibratedClassifierCV(XGBClassifier(n_estimators=n_est), cv=3)

         scores = cross_val_score(model, oof_preds, y, cv=5, scoring="f1_weighted")
         return scores.mean()
     ```

     ---
     Optuna Optimization Summary (PHASE_1B):

     | Stage | Optimization Type | Trials | Parameters | Objective |
     |-------|-------------------|--------|------------|-----------|
     | 7 | Label Optimization | 100 | upper_mult, lower_mult, horizon, atr_period | Sharpe with costs |
     | 8 | Feature Selection | 100 | Binary include/exclude per feature (162 params) | F1 or Sharpe |
     | 9 | Feature Pruning | 50 | importance_threshold, cumulative_importance, correlation_threshold | Performance - complexity |
     | 13 | Hyperparameter Tuning | 100 x 23 models | Model-specific (see above) | F1-weighted |

     Total Optimization Trials:
     - Label trials: 100 (triple-barrier params: upper_mult, lower_mult, horizon, atr_period)
     - Feature selection trials: 100 (binary include/exclude per feature)
     - Feature pruning trials: 50 (importance-based removal)
     - Hyperparameter trials: 100 per model x 23 models = 2,300 trials
     - **TOTAL: ~2,550 optimization trials**

     Optuna Configuration:
     ```python
     OPTUNA_CONFIG = {
         "label_optimization": {
             "n_trials": 100,
             "sampler": "TPESampler",
             "pruner": "MedianPruner",
             "direction": "maximize",
         },
         "feature_selection": {
             "n_trials": 100,
             "sampler": "TPESampler(multivariate=True)",
             "pruner": "SuccessiveHalvingPruner",
             "direction": "maximize",
         },
         "feature_pruning": {
             "n_trials": 50,
             "sampler": "TPESampler",
             "pruner": "NopPruner",
             "direction": "maximize",
         },
         "hyperparameter_tuning": {
             "n_trials": 100,
             "sampler": "TPESampler (boosting/classical) | CmaEsSampler (neural)",
             "pruner": "MedianPruner",
             "direction": "maximize",
         },
     }
     ```                                                                                                 
                                                                                                                                        
     ---                                                                                                                                
     Deliverables                                                                                                                       
                                                                                                                                        
     Create folder /Users/sneh/research/X/ containing 6 phase documents (NO migration phase):                                           
                                                                                                                                        
     1. PHASE_0_FOUNDATION.md - Clean interfaces and data flow                                                                          
     2. PHASE_1_UNIFIED_FEATURES.md - Single feature registry with all 150+ indicators                                                  
     3. PHASE_2_ADAPTER_INTEGRATION.md - Adapters used by Trainer (no bypass)                                                           
     4. PHASE_3_TRAINING_ORCHESTRATION.md - Single entry point for all training                                                         
     5. PHASE_4_META_LEARNERS.md - Heterogeneous ensemble + OOF alignment                                                               
     6. PHASE_5_INFERENCE.md - Auto bundle creation + meta-learner inference                                                            
                                                                                                                                        
     ---                                                                                                                                
     Next Steps                                                                                                                         
                                                                                                                                        
     1. Confirm this inventory captures everything                                                                                      
     2. Create /Users/sneh/research/X/ folder                                                                                           
     3. Write each PHASE_N_*.md with detailed tasks   