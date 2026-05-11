# GPU Baseline v0

## Goal

Run `meta-llama/Llama-3.1-8B-Instruct` on `benchmarks/pagella_v0.jsonl` before any fine-tuning.

## Runtime Options

- Colab: easiest for first manual practice.
- Kaggle: good free alternative, sometimes more friction with secrets.
- RunPod: best if we want a cleaner/professional GPU box.
- Modal: better later, when we want scripted repeatable jobs.

## Prerequisites

- Hugging Face account.
- Access accepted for `meta-llama/Llama-3.1-8B-Instruct`.
- `HF_TOKEN` configured only in the runtime environment.
- No tokens committed to the repo.

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
