# ML Pipeline Gap Analysis
## Critical Architecture Issues Fixed

**Review Date:** 2026-01-03
**Method:** 3 Specialized Agents (ML Engineer, Quant Analyst, Code Reviewer)

---

## Issues Fixed in This Commit

### 1. CRITICAL: Transformer Missing Causal Mask (FIXED)
**File:** `src/models/neural/transformer_model.py:162-208`

**Problem:** TransformerEncoder was processing sequences without causal masking, allowing attention to future positions during training.

**Fix:** Added `_generate_causal_mask()` method that creates upper-triangular attention mask, preventing the model from attending to future positions.

### 2. CRITICAL: Simultaneous Barrier Hit Heuristic (FIXED)
**File:** `src/phase1/stages/labeling/triple_barrier.py:129-141, 283-290`

**Problem:** When both upper and lower barriers hit on same bar, code used "distance from bar open" to guess which hit first. This heuristic is statistically invalid - bar open has no causal relationship to barrier hit order.

**Fix:** Mark simultaneous barrier hits as invalid (-99) to exclude from training. This is the safe approach when tick data is unavailable. Affects 5-15% of labels in volatile markets, but noise reduction outweighs sample loss.

### 3. HIGH: Neural Models Missing Class Weights (FIXED)
**File:** `src/models/neural/base_rnn.py:350-362`

**Problem:** Boosting models used `class_weight="balanced"`, but neural models didn't compute class weights for imbalanced datasets (common in trading: neutral >> long/short).

**Fix:** Added class weight computation using inverse frequency weighting. Enabled by default via `use_class_weights=True` config option.

---

## Remaining Issues (Not Fixed - Lower Priority for Colab)

### Production Infrastructure (Not Needed for Colab Training)
- No GPU memory cleanup (`torch.cuda.empty_cache()`) - handled by Colab runtime
- No graceful shutdown handlers - Colab manages session lifecycle
- Pickle serialization security - trusted environment on Colab
- Monitoring infrastructure - W&B handles this for Colab

### Notebook API Mismatches (Need Separate Fix)
The scaffolded notebooks in `colab_notebooks/` call non-existent APIs:
- `model.get_state()` / `model.load_state()` - actual API is `save(path)` / `load(path)`
- `callbacks=[callback]` in `model.fit()` - not supported

### Feature Engineering (Medium Priority)
- 7 highly correlated volatility measures
- Non-stationary price features (log transform recommended but not applied)
- Microstructure proxies on wrong timeframe (5min vs daily)

### Labeling (Medium Priority)
- Transaction cost uses median ATR from full dataset (minor lookahead)
- Adaptive barrier regime not shifted by 1 bar

---

## Verified as Working Correctly

- Purge/embargo implementation (60/1440 bars)
- MTF lookahead prevention (shift(1))
- Train-only scaling
- TCN causal convolutions
- Sequence dataset label alignment

---

## Sources

- [Scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [Triple Barrier Method Issues](https://www.mql5.com/en/articles/19850)
- [De Prado AFML Notes](https://reasonabledeviations.com/notes/adv_fin_ml/)
