# Qwen2.5 3B v2 Eval-First Guard Runbook

## Goal

Run the current rollback champion `qwen25-3b-fantabrain-sft-v2` on Pagella v0 with an inference-time prompt guard, then audit the predictions for known leakage patterns.

This is an eval-only experiment. Do not train a new adapter, do not edit Pagella v0, and do not create a new dataset from these eval cases.

## 1. GPU Check

```python
!nvidia-smi

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

## 2. Load GitHub Token

Set `GH_TOKEN` as a Colab secret, then expose it to shell cells:

```python
from google.colab import userdata
import os

os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN") or ""
assert os.environ["GH_TOKEN"], "GH_TOKEN missing from Colab Secrets"
```

## 3. Clone Or Update Repo

While this PR is under review, the branch is `codex/v2-eval-first-guard`. After merge, set `FANTABRAIN_BRANCH` to `master`.

```bash
%%bash
set -euo pipefail

BRANCH="${FANTABRAIN_BRANCH:-codex/v2-eval-first-guard}"
REPO_URL="https://${GH_TOKEN}@github.com/paganid86-jpg/fantabrain-llm.git"

if [ ! -d /content/fantabrain-llm/.git ]; then
  git clone "$REPO_URL" /content/fantabrain-llm
fi

cd /content/fantabrain-llm
git fetch origin "$BRANCH"
git switch "$BRANCH"
git pull --ff-only origin "$BRANCH"
git status -sb
```

## 4. Install Dependencies

```bash
%%bash
set -euo pipefail

cd /content/fantabrain-llm
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python -m pip install -U "bitsandbytes>=0.46.1"
```

## 5. Restore Adapter v2

Upload `qwen25-3b-fantabrain-sft-v2-adapter.zip` if the adapter directory is missing.

```python
from google.colab import files
from pathlib import Path
import shutil
import zipfile

%cd /content/fantabrain-llm

target = Path("models/adapters/qwen25-3b-fantabrain-sft-v2")
if not (target / "adapter_config.json").exists():
    uploaded = files.upload()
    zip_name = next(iter(uploaded))
    tmp = Path("/content/fantabrain-upload-adapter-v2")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_name) as archive:
        archive.extractall(tmp)
    source = next(tmp.rglob("qwen25-3b-fantabrain-sft-v2"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

print("adapter config:", (target / "adapter_config.json").exists())
print("adapter model:", any(target.glob("adapter_model.*")))
```

Expected:

```text
adapter config: True
adapter model: True
```

## 6. Guarded Pagella Run

```bash
%%bash
set -euo pipefail

cd /content/fantabrain-llm
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4 \
  --prompt-guard mode_fence_v1
```

Expected output directory:

```text
reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0
```

## 7. Prediction Audit

```bash
%%bash
set -euo pipefail

cd /content/fantabrain-llm
python scripts/audit_predictions.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0 \
  --fail-on-hard-gates
```

The command writes:

```text
prediction_audit.json
prediction_audit.md
```

If `--fail-on-hard-gates` exits with code `1`, the audit still wrote the files. Read `prediction_audit.md` to see which cases triggered hard violations.

## 8. Download Report

```bash
%%bash
set -euo pipefail

cd /content/fantabrain-llm
zip -r qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0.zip \
  reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0
```

```python
from google.colab import files

files.download("/content/fantabrain-llm/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0.zip")
```

## Decision Gate

Keep v2 vanilla as champion unless the guarded run preserves manual score quality and reduces hard audit violations.

Minimum signs worth promoting to deeper scoring:

- no invented module in case 2;
- zero Mantra `modificatore` / `modificatori`;
- fewer Classic role-code and module-language leaks;
- fewer malformed Italian terms than v5;
- manual raw/effective score at least comparable to v2 vanilla.
