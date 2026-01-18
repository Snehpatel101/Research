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
     7. OPTUNA: Label Optimization - 100 trials (barrier params: upper_mult, lower_mult, horizon, atr_period)
     8. OPTUNA: Feature Selection - 100 trials (binary include/exclude)
     9. OPTUNA: Feature Pruning - 50 trials (importance-based removal)
     10. Splits - Train/val/test (70/15/15)
     11. Scaling - Train-only robust scaling
     12. Adaptation - 2D/3D/4D tensor per model type
     13. OPTUNA: Hyperparameter Optimization - 100 trials per model (23 models)
     14. Training - PurgedKFold CV, OOF generation
     15. Stacking - OOF alignment, meta-learner
     16. Bundling - Model + Scaler + Graph -> Artifact

     Optuna Optimization Summary (PHASE_1B):
     - Label trials: 100 (triple-barrier params)
     - Feature selection trials: 100 (binary)
     - Feature pruning trials: 50 (importance-based)
     - Hyperparameter trials: 100 per model
     - Total: ~100 + 100 + 50 + (100 × N_models) trials                                                                                                 
                                                                                                                                        
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