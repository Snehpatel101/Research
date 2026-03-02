# Colab Tomorrow — Full Test Suite Commands

## Quick Reference

```bash
# Clone & setup (run first in Colab)
!git clone https://github.com/Snehpatel101/Research.git /content/research
%cd /content/research
!pip install -q -r requirements-colab.txt
```

---

## Full 14-Model Compatibility Test (skip 4D)

Use this on **Colab T4 (16GB VRAM)**:

```bash
TORCHDYNAMO_DISABLE=1 python scripts/compatibility_test.py --skip-4d
```

---

## Full 17-Model Test (ALL models including PatchTST & iTransformer)

Use this on **Colab A100/V100 (40-80GB VRAM)**:

```bash
python scripts/compatibility_test.py
```

---

## Other Modes

```bash
# Quick sanity check — 2D models only (~45 seconds)
python scripts/compatibility_test.py --quick

# Force CPU (if GPU issues)
python scripts/compatibility_test.py --skip-4d --cpu
```

---

## What's In The Test Suite

| Tests  | Combo          | Models                                                                 |
|--------|----------------|------------------------------------------------------------------------|
| 1-3    | 2D+2D          | xgboost+lightgbm, catboost+rf, logistic+svm                           |
| 4-11   | 2D+3D          | Every 3D model: tcn, lstm, gru, transformer, tft, nbeats, inceptiontime, resnet1d |
| 12-13  | 3D+3D          | tcn+lstm, gru+inceptiontime                                           |
| 14     | 2D+2D+3D       | xgboost+lightgbm+tcn (triple)                                         |
| 15-17  | 4D (full only)  | patchtst, itransformer combos                                         |

---

## What We Know From Local Testing (RTX 4070 Ti, 12GB)

- **223/223 unit tests pass** ✅
- **13/14 compatibility tests pass** on 12GB card
- **TFT** fails OOF generation due to CUDA OOM on 12GB — **will pass on Colab T4/A100** (16-80GB)
- All 2D models ✅, all 3D models except TFT ✅, all 3D+3D ✅, triple combo ✅
- GPU offload fix in place: trained model → CPU before OOF generation
- OOM retry: if OOF still OOMs on GPU, auto-retries on CPU

## Key Changes (commit 38776fd)

- `src/models/device.py` — centralized `release_gpu_memory()` and `offload_model_to_cpu()`
- `src/models/training/training_ops.py` — offload model to CPU BEFORE OOF generation
- `src/models/training/services/oof_generation.py` — OOM retry with CPU fallback
- `src/models/training/artifacts.py` — `_NumpySafeEncoder` for JSON serialization
- `scripts/compatibility_test.py` — 17-test exhaustive model combination suite
- 7 more files updated to use centralized GPU functions
