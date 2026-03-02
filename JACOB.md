# JACOB — TFT on Colab L4: What We're Solving & How to Test

## Current Score: 13/14 ✅ — TFT is the only failure

---

## The Problem

**TFT (Temporal Fusion Transformer) OOMs on Google Colab L4 (23 GB VRAM).**

13 of 14 compatibility tests pass. The one failure is test #8: `2D+3D xgb+tft`.

## Why It Fails — The Chain of Issues

### Issue 1: TFT model is too large for 23 GB (ROOT CAUSE)

TFT with `d_model=256` and `n_features=2490` creates a massive input projection layer:
- `nn.Linear(2490, 256)` = 637,440 params just for the input projection
- Plus LSTM layers, attention layers, gating, variable selection networks
- **Total GPU usage: ~21.3 GB at batch_size=64**
- The L4 has 23 GB, so there's virtually no headroom
- During OOF fold generation, the trained model is STILL on GPU → OOM even at batch=2

### Issue 2: Two "ghost" TFT initializations with DEFAULT config

During the OOF flow, **two extra TFT models get initialized with hardcoded defaults**
(`seq_len=60, batch_size=128, d_model=256`) BEFORE the actual OOF fold training starts.
These consume GPU memory and are never cleaned up:

```
Initialized TFTModel with config: {seq_len=60, batch_size=128, ...}  ← WRONG, should be 30/64
Initialized TFTModel with config: {seq_len=60, batch_size=128, ...}  ← WRONG, ghost #2
Initialized TFTModel with config: {seq_len=30, batch_size=64, ...}   ← CORRECT, actual OOF fold
```

**Where do these come from?** The OOF generation path (`oof_sequence.py` or `oof_generation.py`)
creates model instances that don't receive the trained model's config. We partially fixed this
(commit `4238912` passes `model_config` to `OOFRequest`), but something in the OOF pipeline
still creates 2 extra models with defaults.

### Issue 3 (FIXED): batch_size not reaching ModelTrainingRequest

The experiment config's `batch_size=64` wasn't being passed to `ModelTrainingRequest`, so TFT
fell through to `global.yaml` default (256). **Fixed in commit `5bcec3f`.**

### Issue 4 (FIXED): cuBLAS error with bf16 on L4

TFT + bfloat16 triggered `CUBLAS_STATUS_INTERNAL_ERROR` on L4. We added:
- AMP fallback: bf16 → fp32 with batch size restore (commit `b60cbcd`)
- Extended `is_oom_error()` to catch cuBLAS/cuDNN errors (commit `307d2ed`)

### Issue 5 (JUST PUSHED — UNTESTED ON COLAB): VRAM-adaptive TFT sizing

**Commit `1abbc1e`** makes TFT's architecture scale with VRAM:

| VRAM | d_model | d_ff | n_heads | Expected memory |
|------|---------|------|---------|-----------------|
| ≥80 GB (H100) | 512 | 1024 | 8 | ~40 GB |
| ≥40 GB (A100) | 256 | 512 | 4 | ~21 GB |
| ≥20 GB (L4/3090) | 128 | 256 | 4 | ~6 GB |
| <20 GB (4070 Ti) | 64 | 128 | 2 | ~2 GB |

This should fix the main OOM. **But the ghost initializations (Issue 2) may still waste VRAM.**

---

## How to Test

### On Colab (the target environment):

Open `notebooks/colab_test_runner.ipynb` and connect to an L4 GPU.

**Run cells 1-4** (probe, clone, install, unit tests) then run **cell 12** which:
1. Pulls the latest code (VRAM-adaptive fix)
2. Prints the adaptive TFT config for this GPU
3. Runs all 14 compatibility tests

Expected output for L4 (23 GB):
```
TFT adaptive config: d_model=128, d_ff=256, batch=61, n_heads=4
```

**What PASS looks like:**
```
Total: 14 | Passed: 14 | Failed: 0
```

**What the current FAIL looks like:**
```
[8/14] 2D+3D xgb+tft (xgboost + tft, meta=ridge_meta)
  OOM recovery failed: max retries (6) exceeded
  ❌ FAIL — 100.7s
  No ensemble result — meta-learner training failed
Total: 14 | Passed: 13 | Failed: 1
```

### Locally:

```bash
# Unit tests (should be 223/223)
python -m pytest tests/ -x --tb=short

# Check TFT adaptive config for your GPU
python -c "
from src.models.device import get_optimal_gpu_settings, get_best_gpu
gpu = get_best_gpu()
print(f'GPU: {gpu.name}, VRAM: {gpu.total_memory_gb:.1f} GB' if gpu else 'No GPU')
s = get_optimal_gpu_settings('tft')
print(f'd_model={s[\"d_model\"]}, d_ff={s[\"d_ff\"]}, batch={s[\"batch_size\"]}')
"

# Full compatibility suite (skip 4D models unless you have ≥40GB)
python scripts/compatibility_test.py --skip-4d
```

---

## If TFT Still Fails After VRAM-Adaptive Fix

The remaining suspect is **Issue 2: ghost TFT initializations**. To debug:

1. Run the TFT-only test cell (cell 11) with DEBUG logging
2. Look for lines like `Initialized TFTModel with config: {seq_len=60, batch_size=128}`
3. Those ghost inits come from somewhere in the OOF pipeline creating fresh model instances
   without passing the trained model's config
4. Files to investigate:
   - `src/models/training/services/oof_generation.py` — `_generate_oof_predictions()`
   - `src/models/training/services/oof_sequence.py` — creates the OOF fold requests
   - `src/models/training/training_ops.py` — `_generate_oof()` passes `model_config`

The fix would be: find where those 2 extra TFTModel instances are created and either
(a) don't create them, or (b) pass the trained model's config to them.

---

## Commits (most recent first)

| Commit | What it fixed |
|--------|--------------|
| `d812695` | Notebook updates |
| `1abbc1e` | VRAM-adaptive TFT sizing (d_model scales with GPU) |
| `5bcec3f` | batch_size pass-through to ModelTrainingRequest |
| `4238912` | model_config pass-through to OOFRequest |
| `b60cbcd` | cuBLAS AMP fallback + batch restore |
| `23f7eef` | AMP dtype fallback in base_rnn |
| `307d2ed` | Extended OOM detection for cuBLAS/cuDNN |

## Key Files

| File | What it does |
|------|-------------|
| `src/models/device.py` | VRAM detection, adaptive config, GPU memory mgmt |
| `src/models/neural/tft_model.py` | TFT model — `get_default_config()` now VRAM-aware |
| `src/models/neural/base_rnn.py` | Base neural class — AMP fallback, OOM retry loop |
| `src/models/neural/oom_recovery.py` | OOM detection + batch reduction |
| `src/models/training/training_ops.py` | Training orchestration, OOF config passing |
| `src/models/training/services/oof_generation.py` | OOF fold generation |
| `scripts/compatibility_test.py` | The 14-test suite |
| `notebooks/colab_test_runner.ipynb` | Colab test runner |
