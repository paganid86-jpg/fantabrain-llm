# Qwen2.5 3B LoRA v5 Runbook

## Goal

Train `qwen25-3b-fantabrain-sft-v5` from Dataset v5 and evaluate it on Pagella v0.

Dataset v5 is assembled from `datasets/v2/train.jsonl` plus 16 P4 micro corrective examples. Dataset v3 and Dataset v4 are intentionally not used as a base. Qwen LoRA v2 remains the rollback baseline unless v5 beats the promotion gates. Pagella v0 stays blind: do not copy `benchmarks/pagella_v0.jsonl` into any training file.

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
git fetch origin codex/p4-micro-dataset
git switch codex/p4-micro-dataset
git pull --ff-only origin codex/p4-micro-dataset
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

## 5. Author P4 Draft Blocks

Create these files under `datasets/v5/drafts/` with the exact example counts declared in `datasets/v5/manifest.yaml`:

```text
datasets/v5/drafts/p4_block_001_mantra_no_modificatore.jsonl
datasets/v5/drafts/p4_block_002_classic_clean_vocab.jsonl
datasets/v5/drafts/p4_block_003_no_invented_modules.jsonl
datasets/v5/drafts/p4_block_004_italian_decision_clean.jsonl
```

All rows must follow the project dataset schema, keep Pagella v0 blind, avoid real player names and live facts, and use `source: v5_manual` with `quality_score: 5`.

## 6. Assemble Dataset v5

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/assemble_dataset_v5.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v5/manifest.yaml \
  --output datasets/v5/train.jsonl
```

Expected:

```text
Dataset v5 assembled
```

## 7. Audit Dataset v5

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("datasets/v5/train.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
p4 = [row for row in rows if row.get("source") == "v5_manual"]

print("examples:", len(rows))
print("by mode:", Counter(row.get("mode") for row in rows))
print("by source:", Counter(row.get("source") for row in rows))
print("p4 rows:", len(p4))
print("p4 by mode:", Counter(row.get("mode") for row in p4))
print("min quality:", min(row.get("quality_score", 0) for row in rows))
PY
```

Expected:

```text
examples: 296
by mode: Counter({'mantra': 148, 'classic': 148})
p4 rows: 16
p4 by mode: Counter({'mantra': 8, 'classic': 8})
min quality: 4
```

## 8. Download Dataset v5

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r fantabrain-dataset-v5-296.zip datasets/v5 datasets/v2/train.jsonl datasets/v2/manifest.yaml
```

```python
from google.colab import files

files.download("fantabrain-dataset-v5-296.zip")
```

## 9. Train

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v5.yaml
```

Expected final line:

```text
Adapter saved to models/adapters/qwen25-3b-fantabrain-sft-v5
```

## 10. Download Adapter

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r qwen25-3b-fantabrain-sft-v5-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v5
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v5-adapter.zip")
```

## 11. Evaluate

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v5 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v5-pagella-v0 \
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
head -n 40 reports/runs/qwen25-3b-fantabrain-sft-v5-pagella-v0/summary.json
wc -l reports/runs/qwen25-3b-fantabrain-sft-v5-pagella-v0/predictions.jsonl
zip -r qwen25-3b-fantabrain-sft-v5-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v5-pagella-v0
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v5-pagella-v0.zip")
```
