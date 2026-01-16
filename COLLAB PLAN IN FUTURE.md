# Colab + VS Code + Google Drive (Repo‑Centric, Persistent Results)

This document describes a **robust, repeatable workflow** for using **Google Colab compute** while keeping your project **repo‑centric** and persisting results to **Google Drive**.

This setup is designed for:
- Real repositories (not notebook‑only projects)
- CLI‑driven training / pipelines
- Reproducibility across sessions
- Minimal notebook glue

The notebook is treated as a **runtime launcher**, not the source of truth.

---

## Mental Model (Read This First)

You are working with **three environments**:

1. **Local machine (VS Code)**
   - Where you edit code
   - Where git history lives

2. **Colab runtime (ephemeral)**
   - Where code is executed
   - Has GPU/TPU/CPU
   - Files disappear on restart

3. **Google Drive (persistent storage)**
   - Where results, models, logs live
   - Survives restarts

Key rule:
> **Code lives in the repo (git). Results live in Drive.**

---

## High‑Level Flow

1. Connect VS Code to a Colab runtime
2. Clone your repo into the runtime
3. Install the repo in editable mode
4. Mount Google Drive
5. Configure output paths to point to Drive
6. Run your normal CLI / scripts

---

## Step 1: Open Colab + Connect VS Code

- Install the **Colab VS Code extension**
- Open a `.ipynb` file (this is your launcher notebook)
- Connect to a Colab runtime (GPU if needed)

At this point:
- VS Code edits files locally
- Code execution happens on Colab

---

## Step 2: Clone the Repo Into the Runtime

In the **first notebook cell**:

```bash
# Go to Colab working directory
cd /content

# Clone your repo
git clone https://github.com/<your-org>/<your-repo>.git

cd <your-repo>
```

Notes:
- `/content/<repo>` is the runtime copy
- This folder disappears when the runtime resets
- That is expected

---

## Step 3: Install the Repo (Editable Mode)

This is critical for clean imports.

```bash
pip install -e .
```

Why this matters:
- `import your_package` works everywhere
- Relative imports stop breaking
- CLI entrypoints behave normally

Alternative (not recommended unless necessary):
```bash
export PYTHONPATH=/content/<your-repo>
```

Editable install is cleaner.

---

## Step 4: Mount Google Drive (Persistence Layer)

```python
from google.colab import drive
drive.mount('/content/drive')
```

After mounting:

```text
/content/drive/MyDrive/
```

This directory persists across sessions.

---

## Step 5: Define a Clean Output Layout

Recommended structure inside Drive:

```text
MyDrive/
  <project-name>/
    runs/
      <experiment-name>/
        checkpoints/
        logs/
        metrics/
        artifacts/
```

Example:

```text
/content/drive/MyDrive/topstepx/
  runs/
    mnq_ppo_v3/
      checkpoints/
      logs/
      metrics.json
```

---

## Step 6: Make the Repo Drive‑Aware (Correctly)

### Do NOT hardcode Drive paths everywhere

Instead, use **one configuration layer**:

- Environment variable
- CLI flag
- YAML config

Example (environment variable):

```bash
export OUTPUT_ROOT=/content/drive/MyDrive/topstepx
```

Then inside code:

```python
from pathlib import Path

output_root = Path(os.environ['OUTPUT_ROOT'])
run_dir = output_root / 'runs' / run_name
```

This keeps your repo portable:
- Local → works
- Colab → works
- Cloud → works

---

## Step 7: Fast Scratch vs Persistent Outputs

Use **two tiers of storage**:

### Fast (Ephemeral)
- `/content/<repo>/tmp`
- `/content/cache`

Use for:
- Feature caches
- Temporary artifacts
- Intermediate tensors

### Persistent (Drive)
- `/content/drive/MyDrive/<project>/runs/...`

Use for:
- Model checkpoints
- Final datasets
- Logs, metrics, plots

This avoids Drive I/O slowing training.

---

## Step 8: Running Your Pipeline

Run your **normal CLI commands**:

```bash
python -m cli.train \
  --config configs/mnq.yaml \
  --run-name mnq_ppo_v3 \
  --output-root $OUTPUT_ROOT
```

The notebook should only:
- Set up environment
- Launch commands

No training logic in notebooks.

---

## Step 9: What Happens on Runtime Reset

When Colab disconnects:

- ❌ `/content/<repo>` → deleted
- ❌ `/content/tmp` → deleted
- ✅ `/content/drive/...` → intact

Next session:
1. Reconnect runtime
2. Re‑run setup cells
3. Clone repo again
4. Resume from checkpoints in Drive

This is normal and expected.

---

## Step 10: Resume Runs Cleanly

Because outputs live in Drive:

- Resume from last checkpoint
- Compare runs across sessions
- Share results without copying

Your repo remains stateless.

---

## Common Mistakes (Avoid These)

❌ Treating Drive as the codebase
❌ Writing training logic inside notebooks
❌ Hardcoding `/content/drive/...` paths
❌ Committing large artifacts to git
❌ Expecting runtime disk to persist

---

## Summary (One‑Sentence Rule)

> **Clone the repo into Colab, install it editable, run everything from the repo, and write only results to Drive.**

This gives you:
- Clean architecture
- Reproducibility
- No notebook spaghetti
- Zero dependency on Drive for code

---

If you want, next we can:
- Add a reusable `colab_bootstrap.sh`
- Add a `Makefile` target for Colab runs
- Add automatic resume logic
- Wire this into experiment tracking (W&B / MLflow)

