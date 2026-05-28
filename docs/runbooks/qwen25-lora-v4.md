# Qwen2.5 3B LoRA v4 Runbook

## Goal

Train `qwen25-3b-fantabrain-sft-v4` from Dataset v4 and evaluate it on Pagella v0.

Dataset v4 is assembled from `datasets/v2/train.jsonl` plus 20 P3bis corrective examples. Dataset v3 is intentionally not used as a base. Pagella v0 stays blind: do not copy `benchmarks/pagella_v0.jsonl` into any training file.

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

## 4. Restore Dataset v2

Upload `fantabrain-dataset-v2-280.zip` if `datasets/v2/train.jsonl` is missing.

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

## 5. Author P3bis Draft Blocks

Create these files under `datasets/v4/drafts/` with the exact example counts declared in `datasets/v4/manifest.yaml`:

```text
datasets/v4/drafts/p3bis_block_001_mantra_anti_leak.jsonl
datasets/v4/drafts/p3bis_block_002_classic_anti_leak.jsonl
datasets/v4/drafts/p3bis_block_003_decision_inversion.jsonl
datasets/v4/drafts/p3bis_block_004_refusal_stop.jsonl
```

All rows must follow the project dataset schema, keep Pagella v0 blind, avoid real player names and live facts, and use `source: v4_manual` with `quality_score: 5`.

## 6. Assemble Dataset v4

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/assemble_dataset_v4.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v4/manifest.yaml \
  --output datasets/v4/train.jsonl
```

Expected:

```text
Dataset v4 assembled
```

## 7. Audit Dataset v4

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("datasets/v4/train.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
p3bis = [row for row in rows if row.get("source") == "v4_manual"]

print("examples:", len(rows))
print("by mode:", Counter(row.get("mode") for row in rows))
print("by source:", Counter(row.get("source") for row in rows))
print("p3bis rows:", len(p3bis))
print("p3bis by mode:", Counter(row.get("mode") for row in p3bis))
print("min quality:", min(row.get("quality_score", 0) for row in rows))
PY
```

Expected:

```text
examples: 300
by mode: Counter({'mantra': 150, 'classic': 150})
p3bis rows: 20
p3bis by mode: Counter({'mantra': 10, 'classic': 10})
min quality: 4
```

## 8. Download Dataset v4

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r fantabrain-dataset-v4-300.zip datasets/v4 datasets/v2/train.jsonl datasets/v2/manifest.yaml
```

```python
from google.colab import files

files.download("fantabrain-dataset-v4-300.zip")
```

## 9. Train

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v4.yaml
```

Expected final line:

```text
Adapter saved to models/adapters/qwen25-3b-fantabrain-sft-v4
```

## 10. Download Adapter

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r qwen25-3b-fantabrain-sft-v4-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v4
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v4-adapter.zip")
```

## 11. Evaluate

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v4 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v4-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

## 12. Verify And Download Pagella

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
head -n 40 reports/runs/qwen25-3b-fantabrain-sft-v4-pagella-v0/summary.json
wc -l reports/runs/qwen25-3b-fantabrain-sft-v4-pagella-v0/predictions.jsonl
zip -r qwen25-3b-fantabrain-sft-v4-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v4-pagella-v0
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v4-pagella-v0.zip")
```
