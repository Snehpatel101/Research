Build ONE cohesive ML factory from scratch. NO backward compatibility - clean slate approach.                                      
                                                                                                                                        
     Goal: Single unified pipeline where data flows seamlessly from raw OHLCV to deployed inference bundle.                             
                                                                                                                                        
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
                                                                                                                                        
     150+ Feature Indicators (12 Families)                                                                                              
                                                                                                                                        
     Momentum (7 functions, ~23 features):                                                                                              
     - RSI (14-period + overbought/oversold flags)                                                                                      
     - MACD (12/26/9 + histogram + crossover signals)                                                                                   
     - Stochastic (%K, %D, smoothed)                                                                                                    
     - Williams %R                                                                                                                      
     - ROC (rate of change)                                                                                                             
     - CCI (commodity channel index)                                                                                                    
     - MFI (money flow index)                                                                                                           
                                                                                                                                        
     Moving Averages (4 functions, ~12 features):                                                                                       
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
                                                                                                                                        
     Price Features (4 functions, ~8 features):                                                                                         
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
                                                                                                                                        
     Entropy (5 functions, ~8 features):                                                                                                
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
                                                                                                                                        
     Temporal (4 functions, ~7 features):                                                                                               
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
                                                                                                                                        
     Pipeline Stages (15 total)                                                                                                         
                                                                                                                                        
     1. Ingestion - Load raw OHLCV                                                                                                      
     2. Cleaning - Resample, gap handling                                                                                               
     3. Sessions - Trading hours filtering                                                                                              
     4. MTF Upscaling - 9 timeframes from 1-min                                                                                         
     5. Features - 150+ indicators                                                                                                      
     6. Regime - Market regime detection                                                                                                
     7. Labeling - Triple-barrier labels                                                                                                
     8. GA Optimize - Optuna parameter tuning                                                                                           
     9. Final Labels - Apply optimized params                                                                                           
     10. Splits - Train/val/test (70/15/15)                                                                                             
     11. Scaling - Train-only robust scaling                                                                                            
     12. Datasets - Model-specific formats                                                                                              
     13. Scaled Validation - Data quality                                                                                               
     14. Validation - Feature/label checks                                                                                              
     15. Reporting - Completion reports                                                                                                 
                                                                                                                                        
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