# ML Pipeline Test Coverage Gap Analysis

**Date:** 2025-12-21
**Total Tests:** 715 tests across 16,141 lines
**Codebase:** Ensemble Price Prediction Pipeline with 50+ features, Triple-Barrier Labeling, GA Optimization

---

## Executive Summary

**Overall Test Maturity: 6.5/10** (Good foundation, critical gaps remain)

### Strengths
- Comprehensive stage integration tests (715 total tests)
- Good coverage of core pipeline stages (1-8)
- Excellent leakage prevention tests for FeatureScaler
- Triple-barrier labeling edge cases well tested

### Critical Gaps
1. **Missing unit tests for individual feature calculation functions** (36 functions, ~6 tested)
2. **No ML-specific leakage tests** (fit-on-train-only validation missing)
3. **No temporal integrity tests** (forward-looking bias detection)
4. **Missing GA optimization convergence tests**
5. **No cross-asset feature tests** (MES-MGC correlation, beta)
6. **Insufficient purge/embargo boundary tests**

---

## 1. Missing Unit Tests

### Feature Calculation Functions (CRITICAL GAP)

**Coverage: ~17% (6/36 functions have dedicated unit tests)**

#### TESTED Functions (in test_stage3_feature_engineering_core.py):
- ✓ `calculate_sma_numba` - SMA calculation
- ✓ `calculate_ema_numba` - EMA calculation
- ✓ `calculate_rsi_numba` - RSI bounds and trend behavior
- ✓ `calculate_atr_numba` - ATR positive values
- ✓ MACD components (line, signal, histogram)
- ✓ Bollinger Bands (upper > middle > lower)

#### UNTESTED Functions (High Priority):

**Momentum Features** (`momentum.py`):
- ❌ `add_stochastic()` - Stochastic oscillator %K, %D
  - Missing: Overbought/oversold flag tests (>80, <20)
  - Missing: %K vs %D crossover tests
- ❌ `add_mfi()` - Money Flow Index
  - Missing: Volume-weighted price change tests
  - Missing: Bounds validation (0-100)

**Volatility Features** (`volatility.py`):
- ❌ `add_parkinson_volatility()` - Range-based volatility
  - Missing: High/low ratio calculation validation
  - Missing: Comparison with historical volatility
- ❌ `add_garman_klass_volatility()` - OHLC-based volatility
  - Missing: Formula correctness validation
  - Missing: Edge case tests (gaps, limit moves)

**Volume Features** (`volume.py`):
- ❌ `add_volume_features()` - OBV, volume ratio, z-score
  - Missing: OBV accumulation direction tests
  - Missing: Volume spike detection validation
- ❌ `add_vwap()` - Volume-weighted average price
  - Missing: Session reset tests (daily VWAP resets at midnight)
  - Missing: Price-to-VWAP deviation validation
  - Missing: Zero volume handling

**Trend Features** (`trend.py`):
- ❌ `add_adx()` - Average Directional Index
  - Missing: Trend strength interpretation (ADX > 25)
  - Missing: +DI/-DI crossover tests
- ❌ `add_supertrend()` - Supertrend indicator
  - Missing: ATR multiplier sensitivity tests
  - Missing: Trend reversal signal validation

**Price Features** (`price_features.py`):
- ❌ `add_returns()` - Log returns, simple returns
  - Missing: Return calculation correctness
  - Missing: Compounding validation
- ❌ `add_price_ratios()` - High/low ratio, close/open ratio
  - Missing: Ratio bounds validation
  - Missing: Zero/negative price handling

**Temporal Features** (`temporal.py`):
- ❌ `add_session_features()` - Asia/London/NY session encoding
  - Missing: Session boundary tests (UTC timezone handling)
  - Missing: Overlap period validation
- ❌ `add_temporal_features()` - Hour/minute/day sin/cos encoding
  - Tested in test_stage3_feature_engineering_core.py but incomplete
  - Missing: Cyclical continuity tests (23:59 → 00:00)

**Regime Features** (`regime.py`):
- ❌ `add_trend_regime()` - Trend classification (-1, 0, 1)
  - Missing: Regime transition tests
  - Missing: Threshold sensitivity validation
- ❌ `add_volatility_regime()` - High/low volatility flags
  - Missing: Regime change detection
  - Missing: Percentile-based classification tests

**Cross-Asset Features** (`cross_asset.py`) - **COMPLETELY UNTESTED**:
- ❌ `add_cross_asset_features()` - MES-MGC correlation, beta, spread
  - Missing: Correlation calculation validation (-1 to 1 bounds)
  - Missing: Beta calculation tests (covariance/variance)
  - Missing: Spread z-score normalization
  - Missing: Relative strength (MES return - MGC return)
  - Missing: Handling when only one symbol is present

**Moving Average Features** (`moving_averages.py`):
- ❌ `add_sma()` - Multiple-period SMA
  - Tested via test_stage3 but not comprehensive
  - Missing: Multiple period validation (5, 10, 20, 50, 200)
- ❌ `add_ema()` - Multiple-period EMA
  - Tested via test_stage3 but not comprehensive
  - Missing: EMA vs SMA convergence tests

---

## 2. Missing Integration Tests

### Full Pipeline Stage Transitions (PARTIAL COVERAGE)

**Current State:** Tests exist for individual stages, but end-to-end flow is incomplete.

#### MISSING Tests:

**Stage 1 → Stage 2 → Stage 3 Integration:**
- ❌ Raw data ingestion → resampling → feature engineering
  - Missing: Timestamp alignment validation across stages
  - Missing: Symbol-specific processing verification (MES vs MGC)
  - Missing: Data completeness checks (no dropped rows)

**Stage 3 → Stage 4 → Stage 5 Integration:**
- ❌ Features → triple-barrier labeling → GA optimization
  - Missing: ATR dependency validation (Stage 3 atr_14 required for Stage 4)
  - Missing: Label quality propagation to GA fitness function
  - Missing: Symbol-specific barrier asymmetry (MES: 1.5:1.0, MGC: 1.0:1.0)

**Stage 6 → Stage 7 → Stage 8 Integration:**
- ❌ Final labels → splits → validation
  - Missing: Sample weight tier assignment verification
  - Missing: Purge/embargo application validation
  - Missing: Label distribution preservation across splits

### Configuration Propagation (MISSING):
- ❌ Test that `PURGE_BARS = 60` is used consistently in Stage 7
- ❌ Test that `EMBARGO_BARS = 288` creates proper gaps
- ❌ Test that `LABEL_HORIZONS = [5, 20]` excludes H1 everywhere
- ❌ Test that transaction costs (MES: 0.5 ticks, MGC: 0.3 ticks) are applied in GA

---

## 3. ML-Specific Test Gaps (CRITICAL)

### Leakage Prevention Tests

**Current State:** FeatureScaler has excellent leakage tests, but broader pipeline lacks them.

#### MISSING Tests:

**Fit-Only-On-Train Validation:**
- ❌ **Feature normalization leakage test**
  - Test: Fit scaler on train, transform val/test
  - Verify: Val/test statistics DO NOT match their own mean/std
  - Current: Only FeatureScaler tested, not integrated into pipeline

- ❌ **Feature selection leakage test**
  - Test: If feature selection is added, must use only train data
  - Verify: Selected features based on train performance only

- ❌ **Imputation leakage test**
  - Test: If missing value imputation is added, must use train statistics
  - Verify: Val/test use train-derived fill values

**Label Leakage Tests:**
- ❌ **Purge boundary test** (CRITICAL)
  - Test: Last train sample at index T, purge removes T to T+60
  - Verify: First val sample is at index >= T+61
  - Reason: H20 uses max_bars=60, samples within 60 bars see future labels

- ❌ **Embargo boundary test** (CRITICAL)
  - Test: After purge, embargo creates additional 288-bar gap
  - Verify: No label overlap between train end and val start
  - Reason: Prevents correlated returns from bleeding across splits

- ❌ **Feature-label alignment test**
  - Test: Features at time T should NOT use data from T+1 or later
  - Verify: All rolling windows look backward only
  - Current: test_no_lookahead_in_features exists but limited to subset comparison

**Temporal Integrity Tests:**
- ❌ **Forward-looking bias detection**
  - Test: Calculate features on full dataset vs truncated dataset
  - Verify: Features at time T are identical in both
  - Current: test_no_lookahead_in_features does this but only for 20 features

- ❌ **Time series cross-validation leakage**
  - Test: If using TimeSeriesSplit, verify no future data in folds
  - Missing: No tests for time_series_cv.py module

---

## 4. Edge Case Tests

### Data Quality Edge Cases (PARTIAL COVERAGE)

#### MISSING Tests:

**Empty Data Handling:**
- ❌ **Zero-row DataFrame**
  - Test: Pass empty DataFrame to each stage
  - Expected: Graceful error with clear message
  - Current: Only FeatureScaler tests this

- ❌ **Single-row DataFrame**
  - Test: Pass 1-row DataFrame to rolling window features
  - Expected: All features return NaN, no crash

**NaN Handling:**
- ✓ test_feature_nan_handling exists (injects NaN, verifies dropna)
- ❌ **All-NaN column**
  - Test: Column with all NaN values
  - Expected: Feature calculation skips or fills appropriately
  - Current: Only FeatureScaler tests this

- ❌ **Partial NaN sequences**
  - Test: NaN sequences in middle of data (gaps)
  - Expected: Rolling windows handle gaps correctly

**Extreme Values:**
- ❌ **Price gaps** (limit up/down moves)
  - Test: Simulate 10% overnight gap in futures
  - Expected: ATR captures gap, labels handle correctly

- ❌ **Zero volume bars**
  - Test: Bars with volume=0 (illiquid periods)
  - Expected: Volume features (OBV, VWAP) handle gracefully
  - Current: add_vwap has `if volume.sum() == 0` check but untested

- ❌ **Constant price sequences**
  - Test: 100 bars with identical close price
  - Expected: Returns = 0, volatility = 0, indicators handle
  - Current: FeatureScaler tests constant columns but not feature calculation

**Symbol-Specific Edge Cases:**
- ❌ **Single-symbol processing**
  - Test: Process only MES (no MGC data)
  - Expected: Cross-asset features set to NaN
  - Current: add_cross_asset_features has logic but no test

- ❌ **Misaligned timestamps**
  - Test: MES and MGC have different timestamp ranges
  - Expected: Cross-asset features align correctly or fail gracefully

---

## 5. Triple-Barrier Labeling Tests

### Current Coverage: GOOD (test_stage4_triple_barrier_labeling.py)

**Well-Tested:**
- ✓ Upper barrier hits (profit target)
- ✓ Lower barrier hits (stop loss)
- ✓ Timeout (neutral labels)
- ✓ Same-bar hit resolution (distance from open)
- ✓ ATR scaling validation
- ✓ MAE/MFE calculations
- ✓ Multiple horizons (H5, H20)
- ✓ Asymmetric barriers (k_up=1.5, k_down=1.0)

### MISSING Tests:

**Purge/Embargo Specific:**
- ❌ **Purge boundary validation**
  - Test: Verify purge removes exactly `max_bars` samples
  - For H20 (max_bars=60), last 60 train samples should be purged

- ❌ **Embargo gap validation**
  - Test: Verify embargo creates exactly 288-bar gap
  - Gap should be between purged train end and val start

**Label Quality Scoring:**
- ❌ **Quality tier assignment validation**
  - Test: Top 20% get weight=1.5, middle 60% get 1.0, bottom 20% get 0.5
  - Verify: Quality score formula: `q = (MFE - |MAE|) / (MFE + |MAE|)`
  - Current: test_sample_weight_tiers exists but doesn't test actual quality calculation

**Transaction Cost Impact:**
- ❌ **Transaction cost penalty validation**
  - Test: GA fitness function includes transaction cost
  - For MES: 0.5 ticks = 0.5 * $5 = $2.50 per trade
  - For MGC: 0.3 ticks = 0.3 * $10 = $3.00 per trade
  - Verify: Fitness penalizes labels with profit < transaction cost

**Symbol-Specific Barriers:**
- ❌ **MES asymmetric barrier validation**
  - Test: k_up=1.5, k_down=1.0 for MES
  - Reason: Equity drift compensation
  - Verify: More short labels than symmetric case

- ❌ **MGC symmetric barrier validation**
  - Test: k_up=1.0, k_down=1.0 for MGC
  - Reason: Mean-reverting gold market
  - Verify: Balanced long/short distribution

---

## 6. GA Optimization Tests

### Current Coverage: PARTIAL (test_stage5_ga_optimization.py)

**Well-Tested:**
- ✓ Basic GA execution
- ✓ Parameter bounds validation
- ✓ Fitness function basic operation

### MISSING Tests:

**Convergence Tests:**
- ❌ **Fitness improvement over generations**
  - Test: Run GA for 10 generations
  - Verify: Best fitness in gen 10 > best fitness in gen 1

- ❌ **Parameter convergence validation**
  - Test: Check if k_up, k_down, max_bars converge to stable values
  - Verify: Variance of top 10% solutions decreases over generations

**Fitness Function Validation:**
- ❌ **Signal rate requirement (60% minimum)**
  - Test: Fitness function rejects solutions with <60% directional labels
  - Verify: Solutions with 80% neutral get fitness = -1000

- ❌ **Neutral rate target (20-30%)**
  - Test: Fitness rewards solutions with 20-30% neutral labels
  - Verify: Solutions with 5% neutral get penalty
  - Verify: Solutions with 50% neutral get penalty

- ❌ **Profit factor calculation correctness**
  - Test: Verify profit factor = sum(winning_trades) / sum(losing_trades)
  - Based on: labels (+1/-1), MAE (loss), MFE (gain)
  - Current: Formula exists in stage5_ga_optimize.py but untested

- ❌ **Transaction cost integration**
  - Test: Fitness includes transaction cost penalty
  - Verify: MES uses 0.5 ticks, MGC uses 0.3 ticks
  - Verify: Labels with profit < cost get lower fitness

**Sampling Strategy:**
- ❌ **Contiguous block sampling validation**
  - Test: GA uses contiguous time blocks, not random samples
  - Verify: Temporal order preserved in sample
  - Reason: Random sampling breaks time series structure

**Symbol-Specific Optimization:**
- ❌ **MES asymmetry constraint**
  - Test: GA solution for MES has k_up > k_down
  - Verify: Constraint enforced in DEAP bounds

- ❌ **MGC symmetry constraint**
  - Test: GA solution for MGC has k_up ≈ k_down
  - Verify: Fitness rewards symmetric solutions for MGC

---

## 7. Train/Val/Test Splitting Tests

### Current Coverage: GOOD (test_stage7_data_splitting.py)

**Well-Tested:**
- ✓ Split ratios sum to 1.0
- ✓ Chronological order (train < val < test)
- ✓ No overlap between splits
- ✓ Purge removes samples at boundaries
- ✓ Embargo creates gaps

### MISSING Tests:

**Boundary Precision:**
- ❌ **Exact purge boundary test**
  - Test: With PURGE_BARS=60, last train index should be train_end - 60
  - Verify: Indices train_end-59 to train_end are excluded

- ❌ **Exact embargo boundary test**
  - Test: With EMBARGO_BARS=288, gap should be exactly 288 bars
  - Verify: val_start - train_end_after_purge == 288 + 1

**Label Distribution Preservation:**
- ❌ **Label class balance across splits**
  - Test: Long/short/neutral ratio similar in train/val/test
  - Verify: Each split has >= 20% each class (avoid degenerate splits)

- ❌ **Sample weight distribution**
  - Test: Top/middle/bottom tier weights preserved in splits
  - Verify: Each split has ~20% tier-3, ~60% tier-2, ~20% tier-1

**Symbol Balance:**
- ❌ **Per-symbol split validation**
  - Test: MES and MGC both present in all splits
  - Verify: Each symbol has >= 40% of its data in each split
  - Current: test_per_symbol_splitting exists but basic

**Temporal Coverage:**
- ❌ **Date range validation**
  - Test: Train covers earliest dates, test covers latest
  - Verify: No temporal gaps within splits

- ❌ **Session coverage**
  - Test: Each split covers all trading sessions (Asia/London/NY)
  - Verify: Avoid val/test having only NY session (distribution shift)

---

## 8. Data Validation Tests

### Current Coverage: GOOD (test_stage8_data_validation.py)

**Well-Tested:**
- ✓ Schema validation (required columns)
- ✓ Data type validation
- ✓ Range validation (prices > 0)

### MISSING Tests:

**Statistical Validation:**
- ❌ **Distribution shift detection**
  - Test: Compare train vs val/test distributions
  - Verify: Warn if mean/std shift > 3 standard deviations

- ❌ **Outlier detection**
  - Test: Detect outliers beyond 5 standard deviations
  - Verify: Flag for review, don't auto-remove

**Completeness Validation:**
- ❌ **Required feature validation**
  - Test: All 50+ features present in final dataset
  - Verify: No missing feature columns

- ❌ **Label completeness**
  - Test: All samples have labels for all horizons
  - Verify: No missing label_h5 or label_h20

**Consistency Validation:**
- ❌ **OHLC consistency**
  - Test: high >= max(open, close), low <= min(open, close)
  - Verify: No impossible OHLC bars

- ❌ **Volume consistency**
  - Test: Volume >= 0, no negative volume
  - Verify: Flag zero-volume bars

---

## 9. Test Quality Issues

### Hardcoded Values That Hide Bugs

**Current Issues:**

1. **test_stage3_feature_engineering_core.py**
   ```python
   # Line 42-47: Hardcoded SMA values
   assert np.isclose(sma[2], 2.0)
   assert np.isclose(sma[3], 3.0)
   ```
   **Issue:** Doesn't test formula, just checks output matches expected.
   **Fix:** Add formula validation: `sma[i] = mean(prices[i-period+1:i+1])`

2. **test_stage4_triple_barrier_labeling.py**
   ```python
   # Line 148-150: Same-bar hit uses distance heuristic
   # dist_to_upper = |102 - 104| = 2
   # dist_to_lower = |102 - 96| = 6
   ```
   **Issue:** Test assumes implementation detail (distance from open).
   **Fix:** Test the documented behavior, not implementation.

3. **test_stage7_data_splitting.py**
   ```python
   # Line 139: Hardcoded purge value
   assert metadata['purge_bars'] == 60
   ```
   **Issue:** Test only checks value stored, not actual purging behavior.
   **Fix:** Verify that samples at indices [train_end-59, train_end] are excluded.

### Tests Without Proper Assertions

**Examples:**

1. **test_stage3_feature_engineering_core.py**
   ```python
   # Line 380: Just checks no exception raised
   df_result, report = engineer.engineer_features(df, 'TEST')
   ```
   **Missing:** Verify that NaN rows were actually dropped, check report.

2. **test_stage5_ga_optimization.py**
   ```python
   # Just checks GA runs, doesn't verify fitness improvement
   result = run_ga_optimization(...)
   ```
   **Missing:** Assert that best_fitness > threshold, verify convergence.

### Tests That Don't Fail When They Should

**Examples:**

1. **test_no_lookahead_in_features (Line 286-360)**
   ```python
   # Only compares 20 features, not all 50+
   for col in common_cols[:20]:
   ```
   **Issue:** If lookahead exists in features 21-50, test passes.
   **Fix:** Test ALL features, not just first 20.

2. **test_purge_removes_correct_samples (Line 83-102)**
   ```python
   assert train_idx.max() < expected_train_end_raw
   ```
   **Issue:** Weak assertion - only checks train ends earlier, not exact purge count.
   **Fix:** `assert train_idx.max() == expected_train_end_raw - purge_bars - 1`

---

## 10. Recommended New Tests (Priority Order)

### CRITICAL (Must Have Before Production)

1. **Feature Leakage Test Suite**
   ```python
   # tests/test_feature_leakage.py
   def test_fit_on_train_only():
       """Verify all feature transformations fit on train only."""
       # For EACH feature calculation:
       # 1. Compute features on [train]
       # 2. Compute features on [train + val]
       # 3. Compare feature values at same timestamp in train
       # 4. Assert: values are IDENTICAL
       pass

   def test_no_forward_looking_in_rolling_windows():
       """Verify all rolling windows use past data only."""
       # For rolling window features (SMA, EMA, ATR, etc):
       # 1. Calculate feature at time T using data [0:T]
       # 2. Calculate feature at time T using data [0:T+100]
       # 3. Assert: values at T are identical
       pass
   ```

2. **Purge/Embargo Boundary Tests**
   ```python
   # tests/test_purge_embargo_boundaries.py
   def test_purge_exact_boundary():
       """Verify purge removes exactly max_bars samples."""
       # Given: train_end at index 7000, PURGE_BARS=60
       # Expected: samples 6941-7000 are purged
       # Actual: first val sample should be at index >= 7289 (7000 + 1 + 288 embargo)
       pass

   def test_embargo_exact_gap():
       """Verify embargo creates exactly EMBARGO_BARS gap."""
       # Given: train_end_after_purge at index 6940, EMBARGO_BARS=288
       # Expected: val_start at index 7229 (6940 + 1 + 288)
       pass

   def test_no_label_overlap_at_boundaries():
       """Verify labels don't leak across purge/embargo boundaries."""
       # For H20 (max_bars=60):
       # - Sample at index 6940 (last train) has label based on data up to 7000
       # - Sample at index 7229 (first val) has label based on data starting 7229
       # - These labels are INDEPENDENT (no temporal overlap)
       pass
   ```

3. **Cross-Asset Feature Unit Tests**
   ```python
   # tests/test_cross_asset_features.py
   def test_mes_mgc_correlation_bounds():
       """Verify MES-MGC correlation is between -1 and 1."""
       pass

   def test_beta_calculation():
       """Verify beta = cov(MES, MGC) / var(MGC)."""
       pass

   def test_cross_asset_handles_single_symbol():
       """Verify cross-asset features are NaN when only one symbol present."""
       pass
   ```

### HIGH PRIORITY (Should Have)

4. **Feature Calculation Unit Tests**
   ```python
   # tests/test_feature_calculations.py
   def test_stochastic_oscillator():
       """Test Stochastic %K and %D calculation."""
       pass

   def test_mfi_volume_weighting():
       """Test Money Flow Index uses volume correctly."""
       pass

   def test_vwap_session_reset():
       """Test VWAP resets at session boundaries."""
       pass

   def test_adx_trend_strength():
       """Test ADX correctly measures trend strength."""
       pass

   def test_supertrend_reversals():
       """Test Supertrend signal generation."""
       pass
   ```

5. **GA Optimization Validation Tests**
   ```python
   # tests/test_ga_optimization_validation.py
   def test_fitness_improvement_over_generations():
       """Verify GA fitness improves over generations."""
       pass

   def test_signal_rate_constraint():
       """Verify GA enforces 60% minimum signal rate."""
       pass

   def test_neutral_rate_target():
       """Verify GA targets 20-30% neutral labels."""
       pass

   def test_transaction_cost_penalty():
       """Verify transaction costs reduce fitness."""
       pass
   ```

6. **Edge Case Test Suite**
   ```python
   # tests/test_edge_cases_comprehensive.py
   def test_zero_volume_bars():
       """Test all volume features handle zero volume."""
       pass

   def test_price_gaps():
       """Test limit up/down moves."""
       pass

   def test_constant_price_sequences():
       """Test 100 bars with identical price."""
       pass

   def test_single_symbol_processing():
       """Test pipeline with only MES (no MGC)."""
       pass
   ```

### MEDIUM PRIORITY (Nice to Have)

7. **Integration Test Suite**
   ```python
   # tests/test_full_pipeline_integration.py
   def test_stage1_to_stage8_end_to_end():
       """Test complete pipeline from raw data to validated splits."""
       pass

   def test_symbol_specific_processing():
       """Test MES and MGC processed with correct parameters."""
       pass

   def test_configuration_propagation():
       """Test PURGE_BARS, EMBARGO_BARS, HORIZONS used consistently."""
       pass
   ```

8. **Statistical Validation Tests**
   ```python
   # tests/test_statistical_validation.py
   def test_distribution_shift_detection():
       """Detect when val/test distribution differs from train."""
       pass

   def test_label_distribution_preservation():
       """Verify label balance maintained across splits."""
       pass

   def test_ohlc_consistency():
       """Verify OHLC bars are internally consistent."""
       pass
   ```

---

## 11. Test Execution Plan

### Phase 1: Critical Leakage Prevention (Week 1)
1. Implement `test_feature_leakage.py`
2. Implement `test_purge_embargo_boundaries.py`
3. Fix any leakage issues discovered
4. **Success Criteria:** All features pass lookahead test, purge/embargo boundaries exact

### Phase 2: Feature Unit Tests (Week 2)
1. Implement `test_cross_asset_features.py`
2. Implement `test_feature_calculations.py` (all 36 functions)
3. Fix any calculation bugs discovered
4. **Success Criteria:** 100% of feature functions have unit tests

### Phase 3: ML-Specific Tests (Week 3)
1. Implement `test_ga_optimization_validation.py`
2. Implement `test_edge_cases_comprehensive.py`
3. Fix any edge case bugs discovered
4. **Success Criteria:** GA converges reliably, edge cases handled gracefully

### Phase 4: Integration & Validation (Week 4)
1. Implement `test_full_pipeline_integration.py`
2. Implement `test_statistical_validation.py`
3. Add regression tests for any bugs found
4. **Success Criteria:** End-to-end pipeline test passes, no distribution shifts

---

## 12. Specific Test Recommendations

### Test File: `tests/test_feature_unit_tests.py` (NEW)

**Purpose:** Unit test every feature calculation function for correctness.

**Structure:**
```python
class TestMomentumFeatures:
    def test_rsi_bounds(self):
        """RSI should be between 0 and 100."""

    def test_rsi_uptrend(self):
        """RSI > 70 in strong uptrend."""

    def test_macd_histogram(self):
        """MACD histogram = MACD line - signal line."""

    def test_stochastic_overbought(self):
        """Stochastic %K > 80 in overbought conditions."""

class TestVolatilityFeatures:
    def test_atr_positive(self):
        """ATR is always positive."""

    def test_bollinger_bands_order(self):
        """Upper > Middle > Lower."""

    def test_parkinson_vs_historical_vol(self):
        """Parkinson vol should be lower than historical vol (more efficient)."""

class TestVolumeFeatures:
    def test_obv_accumulation(self):
        """OBV increases on up days, decreases on down days."""

    def test_vwap_session_reset(self):
        """VWAP resets at 00:00 UTC (new session)."""

    def test_vwap_zero_volume_handling(self):
        """VWAP handles zero volume bars gracefully."""

class TestCrossAssetFeatures:
    def test_correlation_bounds(self):
        """MES-MGC correlation between -1 and 1."""

    def test_beta_calculation(self):
        """Beta = cov(MES, MGC) / var(MGC)."""

    def test_single_symbol_nan_handling(self):
        """Cross-asset features are NaN when only one symbol present."""
```

**Estimated Size:** ~800 lines, ~50 test methods

---

### Test File: `tests/test_leakage_prevention.py` (NEW)

**Purpose:** Comprehensive leakage detection across all pipeline stages.

**Structure:**
```python
class TestFeatureLeakage:
    def test_rolling_windows_no_lookahead(self):
        """All rolling windows use only past data."""
        # For each feature with rolling window:
        # - Calculate on full data
        # - Calculate on truncated data
        # - Compare values at same timestamp

    def test_scaling_fit_on_train_only(self):
        """Feature scaling uses only train statistics."""

    def test_feature_selection_train_only(self):
        """Feature selection (if added) uses only train data."""

class TestLabelLeakage:
    def test_purge_prevents_label_leakage(self):
        """Purge removes samples that see future labels."""
        # For H20 (max_bars=60):
        # - Last train sample at T
        # - Its label uses data up to T+60
        # - Purge removes samples T-59 to T
        # - First val sample at T+61 has no overlap

    def test_embargo_prevents_return_correlation(self):
        """Embargo prevents correlated returns across splits."""
        # With 288-bar embargo (~1 day):
        # - Train ends at T
        # - Val starts at T+289
        # - Returns at T and T+289 are uncorrelated

    def test_feature_label_alignment(self):
        """Features at time T use only data [0:T], not [T+1:...]."""

class TestTemporalIntegrity:
    def test_chronological_order_preserved(self):
        """All stages maintain chronological order."""

    def test_no_future_information_in_splits(self):
        """Val and test don't leak information to train."""
```

**Estimated Size:** ~600 lines, ~15 test methods

---

### Test File: `tests/test_purge_embargo_precision.py` (NEW)

**Purpose:** Exact boundary validation for purge and embargo.

**Structure:**
```python
class TestPurgeBoundaries:
    def test_purge_exact_indices(self):
        """Verify exact indices purged."""
        # Given: n=10000, train_ratio=0.7, PURGE_BARS=60
        # Expected: train ends at index 6940 (7000 - 60)
        # Indices 6941-7000 are purged

    def test_purge_for_all_horizons(self):
        """Purge value equals max(max_bars) across all horizons."""
        # For H5 (max_bars=15) and H20 (max_bars=60):
        # PURGE_BARS should be 60 (the maximum)

class TestEmbargoBoundaries:
    def test_embargo_exact_gap(self):
        """Verify exact gap size."""
        # Given: train_end_after_purge=6940, EMBARGO_BARS=288
        # Expected: val_start at 7229 (6940 + 1 + 288)

    def test_embargo_between_val_test(self):
        """Embargo also applied between val and test."""

class TestSplitOverlap:
    def test_no_index_overlap(self):
        """No sample appears in multiple splits."""

    def test_no_temporal_overlap(self):
        """No temporal overlap in label-dependent windows."""
```

**Estimated Size:** ~400 lines, ~8 test methods

---

## 13. Current Test Suite Strengths

### Well-Tested Areas

1. **FeatureScaler** (`test_feature_scaler.py`)
   - Excellent leakage prevention tests
   - Comprehensive edge case handling
   - Persistence (save/load) tested
   - Multiple scaler types validated

2. **Triple-Barrier Labeling** (`test_stage4_triple_barrier_labeling.py`)
   - Upper/lower/timeout cases covered
   - Same-bar hit resolution tested
   - ATR scaling validated
   - MAE/MFE calculations verified

3. **Data Splitting** (`test_stage7_data_splitting.py`)
   - Chronological order enforced
   - No overlap validation
   - Purge and embargo basic tests
   - Per-symbol splitting validated

4. **Basic Feature Calculations** (`test_stage3_feature_engineering_core.py`)
   - SMA, EMA, RSI, ATR basics covered
   - Bollinger Bands ordering tested
   - Temporal features with sin/cos encoding validated

---

## 14. Metrics and Goals

### Current Metrics
- **Total Tests:** 715
- **Test Lines:** 16,141
- **Feature Function Coverage:** ~17% (6/36 functions)
- **ML Leakage Tests:** 1 (FeatureScaler only)
- **Integration Tests:** ~5 (stage transitions)

### Target Metrics (After Improvements)
- **Total Tests:** 1000+ (add ~285 new tests)
- **Feature Function Coverage:** 100% (36/36 functions)
- **ML Leakage Tests:** 15+ (comprehensive suite)
- **Integration Tests:** 20+ (full pipeline coverage)
- **Edge Case Coverage:** 50+ tests

### Success Criteria
1. ✓ Every feature calculation function has >= 3 unit tests
2. ✓ All rolling windows proven to have no lookahead bias
3. ✓ Purge/embargo boundaries mathematically validated
4. ✓ GA optimization convergence tested
5. ✓ Cross-asset features fully tested (MES-MGC interaction)
6. ✓ Zero volume, price gaps, constant prices handled gracefully
7. ✓ End-to-end pipeline test passes with real-world data

---

## 15. Risk Assessment

### HIGH RISK (Untested, Production Impact)
1. **Cross-asset feature calculations** - Used in production but completely untested
2. **Purge boundary precision** - Off-by-one errors could cause label leakage
3. **GA transaction cost penalty** - If wrong, unprofitable strategies selected
4. **VWAP session resets** - Wrong timezone handling = wrong signals

### MEDIUM RISK (Tested But Incomplete)
1. **Feature lookahead bias** - Only 20/50+ features tested
2. **Label quality scoring** - Formula exists but not validated
3. **Symbol-specific barriers** - MES/MGC asymmetry not tested
4. **Embargo gap precision** - Basic test exists, but not exact

### LOW RISK (Well Tested)
1. **FeatureScaler leakage prevention** - Comprehensive tests exist
2. **Triple-barrier basic logic** - Upper/lower/timeout well covered
3. **Data splitting chronology** - Order preservation tested

---

## Conclusion

The test suite has a **solid foundation (715 tests, 16K lines)** but critical gaps remain:

**Top 3 Priorities:**
1. **Feature unit tests** - 30/36 functions untested (83% gap)
2. **Leakage prevention** - Only FeatureScaler tested, broader pipeline untested
3. **Cross-asset features** - 100% untested despite production use

**Recommended Action:**
1. Implement `test_feature_leakage.py` (CRITICAL - Week 1)
2. Implement `test_cross_asset_features.py` (HIGH - Week 2)
3. Implement `test_feature_calculations.py` (HIGH - Week 2-3)
4. Implement `test_purge_embargo_precision.py` (CRITICAL - Week 1)

**Expected Outcome:**
- Test coverage: 6.5/10 → 9.0/10
- Feature unit tests: 17% → 100%
- Leakage tests: 1 → 15+
- Production confidence: Medium → High

The pipeline is **production-ready** for testing environments, but needs these additional tests before deploying to live trading.
