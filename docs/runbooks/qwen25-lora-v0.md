# Qwen2.5 LoRA v0

## Goal

Train the first Qwen2.5 3B LoRA adapter for FantaBrain and evaluate it on Pagella v0.

This run answers one question: does Dataset v0 move Qwen from generic assistant behavior toward
FantaBrain coach behavior?

## Inputs

- Training set: `datasets/v0/train.jsonl`
- Blind eval set: `benchmarks/pagella_v0.jsonl`
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Config: `configs/sft/qwen25-3b-qlora-v0.yaml`
- Adapter output: `models/adapters/qwen25-3b-fantabrain-sft-v0`

Do not train on `benchmarks/pagella_v0.jsonl`. Pagella v0 stays blind.

## Colab Setup

Use a GPU runtime. A Tesla T4 can run this as a learning run with 4-bit loading.

After cloning the repository, switch to the branch that contains the Qwen LoRA recipe:

```bash
git fetch origin codex/qwen25-lora-v0
git switch codex/qwen25-lora-v0
git status -sb
```

Install dependencies:

```bash
python -m pip install --upgrade pip -q
python -m pip install -e ".[train]" -q
python -m pip install -U "bitsandbytes>=0.46.1" -q
python -m pytest
```

## Train Adapter

```bash
python scripts/train_lora.py \
  --config configs/sft/qwen25-3b-qlora-v0.yaml
```

Expected adapter files:

```bash
ls models/adapters/qwen25-3b-fantabrain-sft-v0
test -f models/adapters/qwen25-3b-fantabrain-sft-v0/adapter_config.json
test -f models/adapters/qwen25-3b-fantabrain-sft-v0/adapter_model.safetensors
```

## Inspect Chat Template

```bash
python scripts/inspect_chat_template.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --eval benchmarks/pagella_v0.jsonl \
  --case 1
```

Training should render the full `system -> user -> assistant` example.
Eval should render `system -> user` plus a generation prompt.

## Evaluate Adapter

Use the same decoding defaults from P0:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v0 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v0-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

Verify the summary:

```bash
head -n 60 reports/runs/qwen25-3b-fantabrain-sft-v0-pagella-v0/summary.json
wc -l reports/runs/qwen25-3b-fantabrain-sft-v0-pagella-v0/predictions.jsonl
```

The summary must include:

- `examples: 40`
- `adapter: models/adapters/qwen25-3b-fantabrain-sft-v0`
- `load_in_4bit: true`
- P0 decoding defaults

## Package Results

```bash
zip -r qwen25-3b-fantabrain-sft-v0-pagella-v0.zip \
  reports/runs/qwen25-3b-fantabrain-sft-v0-pagella-v0 \
  models/adapters/qwen25-3b-fantabrain-sft-v0/adapter_config.json
```

Do not package Hugging Face tokens, `.env` files, notebook secrets, or full model weights.

## Compare Against Base

Compare this adapter run against:

`practice-qwen25-3b-pagella-v0-colab-t4-4bit-p0-rerun`

The first signs of progress should be:

- More explicit decisions: `Sceglierei`, `Eviterei`, `Preferirei`
- More grounded refusals when context is missing
- Fewer invented rules, names, numbers, and percentages
- Better Mantra role vocabulary: `E`, `M`, `T`, `W`, `Pc`, `Dc`, `A`
- Shorter, less generic answers

This adapter is not expected to be production-ready. It is a proof-of-learning run.
