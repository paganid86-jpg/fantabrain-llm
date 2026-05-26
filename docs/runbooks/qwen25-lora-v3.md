# Qwen2.5 3B LoRA v3 Runbook

## Goal

Train `qwen25-3b-fantabrain-sft-v3` from Dataset v3 and evaluate it on Pagella v0.

Dataset v3 is assembled from `datasets/v2/train.jsonl` plus the P3 cleanup blocks listed in `datasets/v3/manifest.yaml`. Pagella v0 stays blind: do not copy `benchmarks/pagella_v0.jsonl` into any training file.

## 1. Setup GPU Check

```python
!nvidia-smi

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

## 2. Clone Or Update Repo

Set `GH_TOKEN` as a Colab secret or environment variable before running this cell.

```bash
%%bash
set -euo pipefail

REPO_URL="https://${GH_TOKEN}@github.com/paganid86-jpg/fantabrain-llm.git"

if [ ! -d fantabrain-llm/.git ]; then
  git clone "$REPO_URL" fantabrain-llm
fi

cd fantabrain-llm
git fetch origin codex/p1-dataset-v1
git switch codex/p1-dataset-v1
git pull --ff-only origin codex/p1-dataset-v1
git status -sb
```

## 3. Install Dependencies

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python - <<'PY'
import bitsandbytes

print("bitsandbytes:", bitsandbytes.__version__)
PY
```

## 4. Restore And Check Dataset v2

Make sure `datasets/v2/train.jsonl` exists before assembling Dataset v3. If it is missing in Colab, upload `fantabrain-dataset-v2-280.zip`.

```python
from google.colab import files
from pathlib import Path
import shutil
import zipfile

%cd /content/fantabrain-llm

target = Path("datasets/v2/train.jsonl")
if not target.exists():
    uploaded = files.upload()
    zip_name = next(iter(uploaded))
    tmp = Path("/content/fantabrain-upload-v2")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_name) as archive:
        archive.extractall(tmp)
    source = next(tmp.rglob("datasets/v2/train.jsonl"))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)

print("v2 examples:", sum(1 for _ in target.open(encoding="utf-8")))
```

Expected:

```text
v2 examples: 280
```

## 5. Author P3 Draft Blocks

Create these files under `datasets/v3/drafts/` with the exact example counts declared in `datasets/v3/manifest.yaml`:

```text
datasets/v3/drafts/p3_block_001_mantra_no_module_invention.jsonl
datasets/v3/drafts/p3_block_002_classic_modificatore_clean.jsonl
datasets/v3/drafts/p3_block_003_refusal_stop_clean.jsonl
datasets/v3/drafts/p3_block_004_mantra_roles_no_cross_mode.jsonl
datasets/v3/drafts/p3_block_005_italiano_cleanup_decision_first.jsonl
```

Each row must follow the project dataset schema, keep Pagella v0 blind, avoid real player names and live facts, and use `source: v3_manual`.

## 6. Validate And Assemble Dataset v3

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/assemble_dataset_v3.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v3/manifest.yaml \
  --output datasets/v3/train.jsonl
```

Sanity check the assembled file:

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("datasets/v3/train.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
p3_rows = [row for row in rows if row.get("source") == "v3_manual"]

print("examples:", len(rows))
print("by mode:", Counter(row.get("mode") for row in rows))
print("p3 rows:", len(p3_rows))
print("p3 by mode:", Counter(row.get("mode") for row in p3_rows))
PY
```

Expected:

```text
examples: 320
p3 rows: 40
p3 by mode: Counter({'mantra': 20, 'classic': 20})
```

## 7. Train

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v3.yaml
```

Expected final line:

```text
Adapter saved to models/adapters/qwen25-3b-fantabrain-sft-v3
```

## 8. Evaluate

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v3 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v3-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

## 9. Verify And Download Artifacts

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
head -n 40 reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/summary.json
wc -l reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/predictions.jsonl
zip -r qwen25-3b-fantabrain-sft-v3-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v3
zip -r qwen25-3b-fantabrain-sft-v3-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v3-adapter.zip")
files.download("qwen25-3b-fantabrain-sft-v3-pagella-v0.zip")
```
