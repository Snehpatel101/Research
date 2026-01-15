1.2 Feature Store / Feature Caching                                                                                 
                                                                                                                      
  Impact: High | Effort: 1 week                                                                                       
                                                                                                                      
  Current State: Features recomputed every run (~150 indicators)                                                      
  Problem: 10x computational waste, no consistency guarantee                                                          
                                                                                                                      
  Required:                                                                                                           
  - Parquet-based feature versioning                                                                                  
  - Point-in-time retrieval                                                                                           
  - Feature lineage tracking                                                                                          
  - Checksums for integrity                                                                                           
                                                                                                                      
  New Module: src/feature_store/                                                                                      
                                                                                                                      
  ---                                                                                                                 
  1.3 Production Backtester with Position Sizing                                                                      
                                                                                                                      
  Impact: Critical | Effort: 1 week                                                                                   
                                                                                                                      
  Current State: src/models/metrics.py has simplified metrics (±1 returns proxy)                                      
  Problem: Reported metrics are 30-50% optimistic vs real trading                                                     
                                                                                                                      
  Missing (from metrics.py:167):                                                                                      
  # NOTE: These are simplified trading metrics for quick comparison                                                   
  # For production, integrate with vectorbt or zipline                                                                
                                                                                                                      
  Required:                                                                                                           
  - Actual P&L calculation with transaction costs                                                                     
  - Position sizing (Kelly criterion, fixed fractional)                                                               
  - Slippage modeling                                                                                                 
  - Max drawdown calculation                                                                                          
  - Equity curve simulation                                                                                           
  - Sortino ratio, Calmar ratio


 2.1 Complete MTF Iteration (Stages 3-6)                                                                             
                                                                                                                      
  Impact: High | Effort: 3-5 days                                                                                     
                                                                                                                      
  Current State: MTF upscaling works (9 TFs), but downstream stages only process target_timeframe                     
  Location: src/phase1/stages/mtf/generator.py                                                                        
                                                                                                                      
  Problem: Heterogeneous ensembles can't fully leverage multi-TF diversity                                            
  Required: Iterate feature/labeling/scaling stages across all timeframes   



 2.4 Meta-Labeling for Bet Sizing                                                                                    
                                                                                                                      
  Impact: High | Effort: 3-4 days                                                                                     
                                                                                                                      
  Current State: Triple-barrier labeling only (direction)                                                             
  Industry Best Practice: Two-stage approach (direction + size)                                                       
                                                                                                                      
  Required:                                                                                                           
  - Primary model: High-recall direction classifier                                                                   
  - Meta-model: Confidence-based sizing                                                                               
  - Integration with stacking pipeline    


  3.2 Ensemble Diversity Metrics                                                                                      
                                                                                                                      
  Impact: Medium | Effort: 2 days                                                                                     
                                                                                                                      
  Current State: No diversity measurement or enforcement                                                              
  Location: src/models/ensemble/stacking.py                                                                           
                                                                                                                      
  Required:                                                                                                           
  - Correlation analysis between base predictions                                                                     
  - Q-statistic, disagreement measure                                                                                 
  - KL divergence penalty in training loss (per FinRL research)     


 3.3 Regime-Conditional Evaluation                                                                                   
                                                                                                                      
  Impact: Medium | Effort: 2 days                                                                                     
                                                                                                                      
  Current State: Global metrics only                                                                                  
  Location: src/models/metrics.py                                                                                     
                                                                                                                      
  Required:                                                                                                           
  - Volatility regime breakdown                                                                                       
  - Trend/mean-reversion separation                                                                                   
  - Time-of-day analysis                                                                                              
  - Per-regime Sharpe calculation                                                                                     
                                                                                                                      
  ---                                                                                                                 
  3.4 Missing Features                                                                                                
                                                                                                                      
  Impact: Low-Medium | Effort: 1-2 days                                                                               
                                                                                                                      
  | Feature                            | Priority | Effort  |                                                         
  |------------------------------------|----------|---------|                                                         
  | TWAP (Time-Weighted Average Price) | Medium   | 2 hours |                                                         
  | GARCH volatility forecast          | Low      | 4 hours |                                                         
  | Hurst exponent as base feature     | Low      | 2 hours |                                                         
  | Sample entropy                     | Low      | 2 hours |                                                         
                                                                                                                      
  ---                                                                                                                 
  Priority 4: Nice-to-Have Improvements                                                                               
                                                                                                                      
  | Improvement                 | Impact | Effort  |                                                                  
  |-----------------------------|--------|---------|                                                                                                                                   
  | OOF prediction caching      | Low    | 1 day   |                                                                  
  | Learning rate finder        | Low    | 4 hours |                                                                  
  | Model artifact checksums    | Low    | 2 hours |                                                                                                                                 
  | Real-time streaming adapter | High   | 2 weeks |
