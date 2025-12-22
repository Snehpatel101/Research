# ML Factory - Phase 1 Enhancements TODO

## Goal
Transform Phase 1 from static pipeline to **dynamic ML factory** with user-configurable options.

---

## 1. Session Normalization System

### Session Definitions wrong WE SPLIT UP EVENLY 3 WAYS 24 HOURS NEW YORK STARTS AT 930 AM.
- [ ] Define session boundaries in UTC
  - [ ] New York: 14:30-21:00 UTC (9:30-16:00 ET)
  - [ ] London: 08:00-16:30 UTC (3:00-11:30 ET)
  - [ ] Asia: 23:00-07:00 UTC (18:00-02:00 ET)
- [ ] Handle session overlaps (London+NY)

### Session Filtering
- [ ] Create session filter module
- [ ] Include/exclude sessions via config
- [ ] Session flags as features

### Calendar & Timezone
- [ ] CME holiday calendar integration
- [ ] DST handling (EDT/EST transitions)
- [ ] Partial trading day detection

### Session Normalization
- [ ] Session-specific volatility normalization


---

## 2. Dynamic Horizon Labeling

### Configurable Horizons
- [ ] Replace hardcoded H5/H20 with config list
- [ ] Support horizons: 1, 5, 10, 15, 20, 30, 60, 120 bars
- [ ] Multiple simultaneous horizons

### Timeframe-Aware Scaling
- [ ] 5 bars @ 1m ≠ 5 bars @ 15m
- [ ] Auto-scale horizons when timeframe changes
- [ ] Horizon validation (horizon < data length)

### Purge/Embargo Scaling
- [ ] Auto-scale purge with max horizon
- [ ] Auto-scale embargo with max horizon

---

## 3. Multi-Timeframe (MTF) Upscaling

### Configurable Resampling
- [ ] Replace hardcoded 5m with `TARGET_TIMEFRAME` config
- [ ] Support: 5m, 10m, 15m, 20m, 30m, 45m, 60m
- [ ] OHLCV aggregation rules per timeframe

### Feature Engineering Updates
- [ ] Make feature calculations timeframe-agnostic
- [ ] Indicator periods scale with timeframe
- [ ] Timeframe metadata in outputs

---

## 4. Alternative Labeling Strategies

### Labeling Abstraction
- [ ] Create `LabelingStrategy` base class
- [ ] Strategy pattern for pluggable labelers

### Labeling Types
- [x] Triple-barrier (existing)
- [ ] Directional returns (next N-bar return sign)
- [ ] Threshold labels (+X% before -Y%)
- [ ] Regression targets (actual return value)
- [ ] Meta-labeling (confidence on primary signal)

### Configuration
- [ ] `LABELING_STRATEGY` enum in config
- [ ] Support multiple label types per dataset
- [ ] Label-specific quality metrics

---

## 5. Regime Labeling System

### Regime Detectors
- [ ] Volatility regime (ATR percentile: low/normal/high)
- [ ] Trend regime (ADX + SMA alignment: up/down/sideways)
- [ ] Market structure (Hurst exponent: mean-reverting/trending)
- [ ] Session regime characteristics

### Regime Integration
- [ ] Add regime columns to feature DataFrame
- [ ] Option: Regime as feature (single model)
- [ ] Option: Regime as filter (model per regime)
- [ ] Option: Regime-adaptive barriers

---

## 6. MTF Integration

### Option A: MTF Indicators
- [ ] Calculate indicators on higher TFs (15m, 1h)
- [ ] Align to base TF via forward-fill
- [ ] Features: `rsi_14_15m`, `sma_50_1h`

### Option B: MTF Bars
- [ ] Include OHLCV from higher TFs
- [ ] Timestamp alignment across TFs
- [ ] Features: `close_15m`, `high_1h`

### Validation
- [ ] No lookahead in MTF alignment
- [ ] Handle missing data at TF boundaries

---

## 7. Configuration Updates

### New Config Options
- [ ] `SESSIONS_CONFIG` - Session definitions
- [ ] `HORIZONS` - Dynamic horizon list
- [ ] `TARGET_TIMEFRAME` - Resampling target
- [ ] `LABELING_STRATEGY` - Label type selection
- [ ] `REGIME_CONFIG` - Regime detection settings
- [ ] `MTF_CONFIG` - Multi-timeframe settings

### Presets
- [ ] "scalping" preset (1m, short horizons)
- [ ] "swing" preset (15m-1h, longer horizons)
- [ ] "day_trading" preset (5m, medium horizons)

### Validation
- [ ] Startup validation for all new settings
- [ ] Fail-fast on invalid configurations

---

## File Structure

```
src/
├── config.py                    # MODIFY: Add dynamic configs
├── stages/
│   ├── sessions/                # NEW
│   │   ├── config.py            # Session definitions
│   │   ├── filter.py            # Session filtering
│   │   ├── calendar.py          # Holiday calendar
│   │   └── normalizer.py        # Session normalization
│   ├── labeling/                # NEW
│   │   ├── base.py              # LabelingStrategy base
│   │   ├── triple_barrier.py    # Existing logic
│   │   ├── directional.py       # Return direction
│   │   ├── threshold.py         # Threshold events
│   │   └── regression.py        # Regression targets
│   ├── resampler.py             # NEW: MTF resampling
│   ├── regime_detection.py      # NEW: Regime detectors
│   └── mtf_features.py          # NEW: MTF integration
```

---

## Priority Order

| Priority | Component | Complexity |
|----------|-----------|------------|
| 1 | MTF Upscaling | Medium |
| 2 | Dynamic Horizons | Low |
| 3 | Session Normalization | Medium |
| 4 | Alternative Labeling | Medium |
| 5 | Regime Detection | High |
| 6 | MTF Integration | High |

---

## Success Criteria

- [ ] User selects timeframe via config
- [ ] User selects horizons via config
- [ ] User filters by trading session
- [ ] User chooses labeling strategy
- [ ] Pipeline adapts without code changes
- [ ] All options validated at startup
- [ ] Backward compatible with current workflow

---

## Notes

- All files must stay under 650 lines
- Validate inputs at boundaries (fail-fast)
- No exception swallowing
- Test each component independently
- Maintain backward compatibility
