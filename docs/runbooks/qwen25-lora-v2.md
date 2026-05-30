# Qwen2.5 3B LoRA v2 Runbook

## Goal

Train `qwen25-3b-fantabrain-sft-v2` from Dataset v2 and evaluate it on Pagella v0.

Dataset v2 is assembled from `datasets/v1/train.jsonl` plus the P2 repair blocks listed in `datasets/v2/manifest.yaml`. Pagella v0 stays blind: do not copy `benchmarks/pagella_v0.jsonl` into any training file.

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

## 4. Restore And Check Dataset v1

Make sure `datasets/v1/train.jsonl` exists before assembling Dataset v2.

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
test -f datasets/v1/train.jsonl
python - <<'PY'
import json
from pathlib import Path

path = Path("datasets/v1/train.jsonl")
assert path.exists(), f"missing {path}"

with path.open(encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

print("v1 examples:", len(rows))
for index, row in enumerate(rows[:3], start=1):
    print(f"example {index}: mode={row.get('mode')} task={row.get('task')}")
PY
```

## 5. Author P2 Draft Blocks

Create these files under `datasets/v2/drafts/` with the exact example counts declared in `datasets/v2/manifest.yaml`:

```text
datasets/v2/drafts/p2_block_001_classic_modificatore.jsonl
datasets/v2/drafts/p2_block_002_mantra_role_codes_guardrail.jsonl
datasets/v2/drafts/p2_block_003_risk_varianza_decisioni.jsonl
datasets/v2/drafts/p2_block_004_refusal_grounded_clean.jsonl
datasets/v2/drafts/p2_block_005_italiano_asciutto_decision_first.jsonl
```

Each row must follow the project dataset schema, keep Pagella v0 blind, avoid real player names and live facts, and use `mode: mantra` or `mode: classic`.

## 6. Validate And Assemble Dataset v2

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/assemble_dataset_v2.py \
  --base datasets/v1/train.jsonl \
  --manifest datasets/v2/manifest.yaml \
  --output datasets/v2/train.jsonl
```

Sanity check the assembled file:

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
test -f datasets/v2/train.jsonl
wc -l datasets/v1/train.jsonl datasets/v2/train.jsonl
python -m pytest tests/test_dataset_v2_manifest.py tests/test_assemble_dataset_v2.py -q
```

## 7. Train Adapter

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v2.yaml
```

Expected adapter path:

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
ls -la models/adapters/qwen25-3b-fantabrain-sft-v2
test -f models/adapters/qwen25-3b-fantabrain-sft-v2/adapter_config.json
test -f models/adapters/qwen25-3b-fantabrain-sft-v2/adapter_model.safetensors
```

## 8. Evaluate On Pagella v0

Use the Qwen base model with the v2 adapter and the same decoding values used for the v1/P0 runs.

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

Verify the report:

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
head -n 80 reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/summary.json
wc -l reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl
```

## 9. Zip And Download Adapter And Pagella

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r qwen25-3b-fantabrain-sft-v2-adapter.zip \
  models/adapters/qwen25-3b-fantabrain-sft-v2

zip -r qwen25-3b-fantabrain-sft-v2-pagella-v0.zip \
  reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0 \
  models/adapters/qwen25-3b-fantabrain-sft-v2/adapter_config.json
```

```python
from google.colab import files

files.download("fantabrain-llm/qwen25-3b-fantabrain-sft-v2-adapter.zip")
files.download("fantabrain-llm/qwen25-3b-fantabrain-sft-v2-pagella-v0.zip")
```

Do not package Hugging Face tokens, `.env` files, notebook secrets, or full base model weights.
