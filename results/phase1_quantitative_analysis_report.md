# Phase 1 Ultra-Deep Quantitative Analysis Report

## Ensemble Price Prediction Pipeline for MES/MGC Futures

**Report Date:** December 19, 2025
**Analysis Period:** December 2008 - July 2025
**Total Samples:** ~2.4M (MES: 1.19M, MGC: 1.21M)

---

## Executive Summary

This report presents a critical quantitative analysis of Phase 1 of the ensemble price prediction pipeline. The analysis reveals **several significant issues** that require attention before proceeding to Phase 2:

1. **Label Distribution Imbalance**: Neutral labels are severely underrepresented (0.2-6% vs. target 30-35%)
2. **Barrier Parameters Misaligned**: Current k values differ significantly from empirically optimal values
3. **Backtest Performance Inflated**: Raw Sharpe ratios of 10-27 are unrealistic and mask negative net performance
4. **Transaction Costs Devastating**: Short horizons become completely unprofitable after costs
5. **Strong Serial Correlation**: Labels show significant autocorrelation, indicating information leakage risk

**Bottom Line**: The labels show genuine predictive signal (positive raw win rates), but the current configuration is not suitable for ML training without substantial modifications.

---

## 1. Label Distribution Analysis

### 1.1 Current Distribution vs. Target

| Horizon | Symbol | Long | Short | Neutral | Target Neutral |
|---------|--------|------|-------|---------|----------------|
| H1 | MES | 52.2% | 42.2% | **5.5%** | 30-35% |
| H1 | MGC | 51.5% | 44.3% | **4.2%** | 30-35% |
| H5 | MES | 50.7% | 47.4% | **1.9%** | 30-35% |
| H5 | MGC | 50.9% | 48.3% | **0.8%** | 30-35% |
| H20 | MES | 50.8% | 48.6% | **0.6%** | 30-35% |
| H20 | MGC | 50.6% | 49.2% | **0.2%** | 30-35% |

### 1.2 Critical Issues

**Problem 1: Insufficient Neutral Labels**
- Current neutral percentages (0.2% - 5.5%) are dramatically below the 30-35% target
- This forces the model to make directional predictions even in sideways markets
- For ML training, balanced classes typically outperform imbalanced ones

**Problem 2: Persistent Long Bias**
- All horizons show consistent 2-8% long bias
- This bias persists across years (2008-2025) and time of day
- Hour 13 (1 PM) shows extreme +8.2% long bias
- Low volatility regimes show higher long bias (+13.4% in Low ATR for H1)

**Problem 3: Barriers Too Tight**
- The barrier_analysis.json shows that current parameters (k=0.3, 0.5, 0.75) produce:
  - H1 with k=0.3: ~60% hit rate (40% neutral) - but actual is 5% neutral
  - H5 with k=0.5: ~90% hit rate (10% neutral) - actual is 2% neutral
  - H20 with k=0.75: ~98% hit rate (2% neutral) - actual is 0.6% neutral

### 1.3 Train/Val/Test Distribution Stability

| Split | Period | Long (H5) | Short (H5) | Neutral (H5) |
|-------|--------|-----------|------------|--------------|
| Train | 2008-2020 | 50.6% | 47.4% | 2.0% |
| Val | 2020-2022 | 51.2% | 47.0% | 1.7% |
| Test | 2022-2025 | 50.6% | 47.5% | 1.9% |

**Finding**: Distribution is stable across splits, indicating no significant temporal drift. However, the consistent long bias may reflect the overall bullish market trend since 2008.

---

## 2. Barrier Parameter Validation

### 2.1 Empirically Optimal k Values

Based on the barrier_analysis.json, the optimal k values for ~35% neutral (65% hit rate) are:

| Horizon | Current k | Optimal k (65% hit) | Difference |
|---------|-----------|---------------------|------------|
| H1 | 0.3 | **0.25** | Too wide |
| H5 | 0.5 | **0.9** | Too narrow |
| H20 | 0.75 | **2.0** | Too narrow |

### 2.2 Move-to-ATR Ratio Analysis

| Horizon | Mean Absolute Move | Mean ATR | Move/ATR Ratio |
|---------|-------------------|----------|----------------|
| H1 (1 bar) | 1.04 pts | 2.13 pts | 0.49 |
| H5 (5 bars) | 3.01 pts | 2.13 pts | 1.41 |
| H20 (20 bars) | 6.84 pts | 2.13 pts | 3.21 |

**Key Insight**: The mean move over 20 bars is 3.21x ATR. Using k=0.75 means barriers at only 0.75 ATR, which will almost always be hit. This explains the near-zero neutral percentage for H20.

### 2.3 Recommended Barrier Recalibration

To achieve ~30-35% neutral labels:

| Horizon | k_up | k_down | max_bars | Expected Neutral |
|---------|------|--------|----------|------------------|
| H1 | 0.25 | 0.25 | 5 | ~35% |
| H5 | 0.9 | 0.9 | 15 | ~33% |
| H20 | 2.0 | 2.0 | 60 | ~30% |

### 2.4 Asymmetric Barriers Consideration

The data shows a consistent long bias. Consider asymmetric barriers:
- **Hypothesis**: Market tends to rise faster than it falls (stairs up, elevator down)
- **Test**: Different k_up vs k_down could balance long/short labels
- **Current**: Both MES and MGC show ~5% more longs than shorts across all horizons

---

## 3. Backtest Reality Check

### 3.1 Raw vs. Net Performance

The `final_validation_report.json` reveals a **critical discrepancy** between raw and net performance:

#### MES Performance

| Horizon | Raw Win Rate | Net Win Rate | Raw Sharpe | Net Sharpe | Raw Return | Net Return |
|---------|--------------|--------------|------------|------------|------------|------------|
| H1 | 50.2% | 12.8% | 19.7 | **-178.3** | +1,921% | **-100%** |
| H5 | 60.6% | 32.8% | 21.6 | **-18.3** | +25,752% | **-99.1%** |
| H20 | 61.3% | 47.2% | 9.8 | **0.15** | +2,107% | **+2.3%** |

#### MGC Performance

| Horizon | Raw Win Rate | Net Win Rate | Raw Sharpe | Net Sharpe | Raw Return | Net Return |
|---------|--------------|--------------|------------|------------|------------|------------|
| H1 | 52.3% | 16.4% | 26.7 | **-184.7** | +5,624% | **-100%** |
| H5 | 61.4% | 38.4% | 26.3 | **-16.3** | +78,467% | **-98.4%** |
| H20 | 61.4% | 50.1% | 11.9 | **1.74** | +4,397% | **+71.7%** |

### 3.2 Why Raw Sharpe of 10-27 is Misleading

**Problem 1: Transaction Cost Assumptions**
- Commission: 0.02% round-trip (0.0001 per side)
- Average trade bps before costs: ~0.4-2.5 bps
- Average trade bps after costs: **-1.5 to -3.6 bps**
- The raw profits are smaller than transaction costs for short horizons

**Problem 2: Trade Frequency Impact**
- H1: 75,000-80,000 trades over test period
- H5: 25,000-27,000 trades
- H20: 7,600-8,200 trades
- Higher frequency = more commission drag

**Problem 3: Realistic Sharpe Expectations**
According to [QuantStart](https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement/):
- Retail algorithmic traders: Sharpe > 2 is "very good"
- Quantitative hedge funds: Sharpe > 2 is minimum threshold
- Professional trading desks: Sharpe 1-2 is realistic
- A Sharpe of 27 would make this the best trading strategy in history

### 3.3 What Realistic Performance Looks Like

After transaction costs, only H20 shows any viability:

| Symbol | Horizon | Net Sharpe | Net Profit Factor | Net Max DD | Realistic? |
|--------|---------|------------|-------------------|------------|------------|
| MGC | H20 | 1.74 | 1.09 | 20.2% | Marginal |
| MES | H20 | 0.15 | 1.01 | 29.6% | Marginal |
| All | H1/H5 | Negative | < 1.0 | ~100% | No |

**Verdict**: Only H20 shows marginally positive net performance. H1 and H5 are completely destroyed by transaction costs.

---

## 4. Statistical Validation

### 4.1 Label Autocorrelation

| Horizon | Lag 1 | Lag 5 | Lag 10 | Lag 20 |
|---------|-------|-------|--------|--------|
| H1 | **0.147** | 0.004 | 0.002 | 0.003 |
| H5 | **0.357** | 0.040 | 0.003 | -0.001 |
| H20 | **0.500** | 0.116 | 0.027 | 0.003 |

**Critical Finding**: Strong positive autocorrelation at lag 1, especially for longer horizons:
- H20 has 0.50 autocorrelation - meaning if current label is +1, there's a 75% chance next label is also +1
- This is expected (overlapping horizons) but creates **serious leakage risk** in CV

### 4.2 Label Transition Matrix (H5)

| From/To | Short | Neutral | Long |
|---------|-------|---------|------|
| Short | 66.4% | 0.7% | 32.9% |
| Neutral | 18.5% | 63.2% | 18.3% |
| Long | 30.7% | 0.8% | 68.5% |

**Issue**: Labels are highly persistent. A "long" label has 68.5% probability of being followed by another "long" label. This violates IID assumptions in standard cross-validation.

### 4.3 Bars to Hit Distribution

| Horizon | Mean | Median | 10th %ile | 90th %ile | Max |
|---------|------|--------|-----------|-----------|-----|
| H1 | 1.26 | 1.0 | 1 | 2 | 3 |
| H5 | 2.30 | 2.0 | 1 | 5 | 12 |
| H20 | 3.95 | 3.0 | 1 | 8 | 40 |

**Observation**: Most barriers hit quickly (median of 1-3 bars). This is consistent with barriers being too tight.

### 4.4 Quality Score Distribution

| Horizon | Mean Quality | % Quality > 0.5 |
|---------|--------------|-----------------|
| H1 | 0.627 | 81.5% |
| H5 | 0.554 | 66.2% |
| H20 | 0.515 | 56.7% |

**Finding**: Quality scores are reasonable, with majority of samples exceeding the 0.5 threshold used in backtesting.

---

## 5. Regime Analysis

### 5.1 Volatility Regime Performance

| ATR Regime | Long (H1) | Short (H1) | Neutral (H1) | Long Bias |
|------------|-----------|------------|--------------|-----------|
| Low (< 0.95) | 53.6% | 40.2% | 6.2% | +13.4% |
| Mid (0.95-2.02) | 51.7% | 43.0% | 5.4% | +8.7% |
| High (> 2.02) | 51.5% | 43.5% | 5.0% | +8.0% |

**Finding**: Long bias is most pronounced in low volatility regimes. This may reflect:
1. Bullish drift in calm markets
2. Stop-loss asymmetry (downside moves often more violent)
3. Or simply the underlying asset's bullish trend since 2008

### 5.2 Time-of-Day Effects

Extreme hourly biases detected:
- Hour 13 (1 PM ET): +8.2% long bias
- Hour 14 (2 PM ET): +5.6% long bias
- Hour 05 (5 AM ET): +5.2% long bias (European open)

**Recommendation**: Consider session-aware labeling or time-of-day features to capture these patterns.

### 5.3 Year-over-Year Stability

Annual distributions show remarkable stability (50-52% long, 46-48% short, 1-2% neutral across 2008-2025). No significant regime shifts detected.

---

## 6. Market Microstructure Considerations

### 6.1 5-Minute Bar Appropriateness

**For MES (Micro E-mini S&P 500):**
- Tick size: 0.25 points ($1.25/tick)
- Current ATR (5-min): ~2.13 points (~8.5 ticks)
- Mean 1-bar move: 1.04 points (~4 ticks)

**Assessment**: 5-minute bars are appropriate for MES. Sufficient movement relative to tick size.

**For MGC (Micro Gold):**
- Tick size: 0.10 points ($1.00/tick)
- Current ATR (5-min): ~1.42 points (~14 ticks)
- Mean 1-bar move: 0.70 points (~7 ticks)

**Assessment**: 5-minute bars are appropriate for MGC.

### 6.2 Trading Session Considerations

Current implementation uses 3 equal 8-hour blocks:
- Asia: 18:00 - 02:00 ET
- London: 02:00 - 10:00 ET
- NY: 10:00 - 18:00 ET

**Issue**: Hourly analysis shows varying label distributions by hour, suggesting potential value in session-specific models.

### 6.3 Contract Roll Handling

Not explicitly analyzed, but the 17-year dataset (2008-2025) includes ~68 contract rolls for quarterly futures. Roll dates may introduce noise around expiration.

---

## 7. Feature Analysis

### 7.1 Selected Features (38 total)

From `feature_selection_report.json`:
- Trend: sma_10, close_to_sma_*, macd, macd_hist, roc_*
- Volatility: atr_7, atr_7_pct, bb_width, bb_position
- Momentum: rsi, stoch_k, stoch_d, adx, plus_di, minus_di
- Volume: obv, volume_ratio, volume_zscore
- Time: hour_sin, hour_cos, dow_sin, dow_cos, is_rth
- Regime: vol_regime, trend_regime

### 7.2 Removed Features (22 total)

Removed due to high correlation (> 0.95):
- Multiple SMAs/EMAs (redundant with sma_10)
- ATR variants (atr_14, atr_21 redundant with atr_7)
- Williams %R (redundant with stoch_k)

### 7.3 Feature Concerns

1. **No lag/momentum features**: All features are concurrent. Consider adding lagged returns.
2. **No volatility forecast**: Vol_regime is categorical. Consider GARCH-based volatility forecasts.
3. **No order flow**: OBV is included but no bid-ask spread or volume imbalance features.

---

## 8. Cross-Validation Concerns

### 8.1 Current Implementation

The pipeline uses 70/15/15 train/val/test splits with:
- Purge bars: 20
- Embargo bars: 288 (1 day of 5-min bars)

### 8.2 Issues Identified

**Problem 1: Label Overlap**
According to [Lopez de Prado's work](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/), triple-barrier labels have event times that extend into the future. With max_bars up to 60, labels can overlap by up to 60 bars.

Current purge of 20 bars is **insufficient** for H20 (max_bars=60).

**Problem 2: Serial Correlation**
With label autocorrelation of 0.50 at lag 1 for H20, standard k-fold CV would have severe leakage. The temporal split is appropriate, but walk-forward validation would be better.

**Recommendation**: Implement [Combinatorial Purged Cross-Validation (CPCV)](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method) with:
- Purge period = max_bars (5, 15, or 60 depending on horizon)
- Embargo period = 288 bars (current)
- Multiple test folds for distribution of performance estimates

---

## 9. Recommendations for Phase 2

### 9.1 Critical Fixes (Must Do)

1. **Recalibrate Barrier Parameters**
   ```python
   BARRIER_PARAMS = {
       1: {'k_up': 0.25, 'k_down': 0.25, 'max_bars': 5},   # Target 35% neutral
       5: {'k_up': 0.90, 'k_down': 0.90, 'max_bars': 15},  # Target 33% neutral
       20: {'k_up': 2.00, 'k_down': 2.00, 'max_bars': 60}  # Target 30% neutral
   }
   ```

2. **Increase Purge Bars**
   - H1: purge_bars = 5 (matches max_bars)
   - H5: purge_bars = 15
   - H20: purge_bars = 60

3. **Drop H1 Horizon**
   - Transaction costs completely destroy H1 profitability
   - Consider only H5 and H20, or increase H1 to 3-bar (15 min)

4. **Realistic Transaction Cost Model**
   - Include slippage: 0.5-1 tick per trade
   - Include market impact for larger positions
   - MES: ~$2.50 round-trip commission
   - MGC: ~$2.50 round-trip commission

### 9.2 Important Improvements (Should Do)

5. **Address Long Bias**
   - Test asymmetric barriers (k_down > k_up) to balance classes
   - Or use class weights in ML training

6. **Implement Walk-Forward Validation**
   - Use expanding or rolling window
   - Retrain monthly or quarterly
   - This addresses non-stationarity concerns

7. **Add Regime-Adaptive Barriers**
   - Higher k values in high-volatility regimes
   - Lower k values in low-volatility regimes
   - This follows best practices from [recent research](https://www.mdpi.com/2227-7390/12/5/780)

### 9.3 Nice to Have

8. **Session-Specific Models**
   - Train separate models for Asia/London/NY sessions
   - Or add session indicator as feature

9. **Feature Enhancements**
   - Add lagged returns (1, 5, 20 bars)
   - Add realized volatility forecasts
   - Consider microstructure features if tick data available

10. **Ensemble Architecture**
    - Multi-horizon ensemble (H5 + H20)
    - Multi-symbol ensemble (MES + MGC)
    - Symbol-specific barrier calibration

---

## 10. Expected Realistic Performance

### 10.1 After Recommended Fixes

| Scenario | Expected Sharpe | Expected Win Rate | Expected Max DD |
|----------|-----------------|-------------------|-----------------|
| H20 Only, After Costs | 0.5 - 1.5 | 52-58% | 15-25% |
| H5 Only, After Costs | 0.2 - 0.8 | 50-55% | 20-30% |
| H5+H20 Ensemble | 0.8 - 1.5 | 53-57% | 15-25% |

### 10.2 Reality Check

Per [industry benchmarks](https://www.quantifiedstrategies.com/sharpe-ratio/):
- Sharpe > 0.75 is "good"
- Sharpe > 1.5 is "excellent" and rare
- Sharpe > 2.0 typically indicates overfitting or insufficient OOS data

The raw Sharpe of 10-27 seen in this analysis is **not achievable in live trading**. A realistic target after Phase 2 would be:
- **Sharpe: 0.8 - 1.5**
- **Win Rate: 52-58%**
- **Profit Factor: 1.1 - 1.3**
- **Max Drawdown: 15-25%**

---

## 11. Conclusion

Phase 1 has successfully established a data pipeline and demonstrated that the labels contain genuine predictive signal. However, several critical issues prevent this from being production-ready:

1. **Barrier calibration is wrong** - Need to recalibrate for 30-35% neutral
2. **Short horizons are unprofitable** - Transaction costs exceed edge
3. **Backtest results are misleading** - Raw metrics hide net losses
4. **Cross-validation needs improvement** - Insufficient purging for label overlap

The recommended path forward:
1. Fix barrier parameters using the empirically optimal k values
2. Focus on H20 (and possibly H5) rather than H1
3. Implement proper walk-forward validation with adequate purging
4. Set realistic performance expectations (Sharpe 0.8-1.5, not 10-27)

---

## Sources

- [Triple Barrier Labeling Research (MDPI 2024)](https://www.mdpi.com/2227-7390/12/5/780)
- [Stock Prediction with Triple Barrier (arXiv 2025)](https://arxiv.org/html/2504.02249v2)
- [Sharpe Ratio Benchmarks (QuantStart)](https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement/)
- [Realistic Sharpe Ratios (HighStrike 2025)](https://highstrike.com/what-is-a-good-sharpe-ratio/)
- [Walk-Forward Validation Best Practices (Medium)](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1)
- [Purging and Embargo in CV (QuantInsti)](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
- [CPCV Method (Towards AI)](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)
- [Micro E-mini S&P 500 Specifications (CME Group)](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.html)

---

*Report generated by quantitative analysis pipeline. All metrics computed from /Users/sneh/research/data/final/ and /Users/sneh/research/results/.*
