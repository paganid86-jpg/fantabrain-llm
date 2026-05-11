# Baseline Eval Harness Design

## Goal

Create a repeatable harness that runs a base model against Pagella v0 and saves model predictions beside the ideal answers.

## Scope

The harness should:

- read `benchmarks/pagella_v0.jsonl`;
- call a pluggable chat client;
- write `predictions.jsonl`, `comparison.md`, and `summary.json`;
- support a local `echo` provider for tests and dry runs;
- support a `transformers` provider for Colab, Kaggle, RunPod, or another Linux GPU runtime;
- support an `openai-compatible` provider for future vLLM, RunPod serverless, Modal, or similar endpoints.

The first production-quality measurement is still manual review. This harness creates the raw material for that review; it does not judge model quality automatically.

## Architecture

The implementation separates generation from provider-specific inference:

- `src/fantabrain_llm/inference.py` owns provider clients.
- `src/fantabrain_llm/predictions.py` owns prediction record rendering and report writing.
- `scripts/generate_predictions.py` is a thin CLI wrapper.

Provider dependencies are loaded lazily. Local tests must pass without installing `torch`, `transformers`, or any OpenAI SDK.

## Output Contract

For each eval row, `predictions.jsonl` contains:

- `case_id`;
- `mode`;
- `task`;
- `tags`;
- `prompt`;
- `expected`;
- `prediction`;
- `provider`;
- `model`;

`comparison.md` renders each case with prompt, prediction, expected answer, and blank manual score fields.

`summary.json` records run name, model, provider, eval path, example count, and timestamp.

## Default Command

Local smoke run:

```powershell
python scripts/generate_predictions.py `
  --provider echo `
  --model echo-baseline `
  --eval benchmarks/pagella_v0.jsonl `
  --run-name echo-pagella-v0-smoke
```

GPU run:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name base-llama31-pagella-v0
```

OpenAI-compatible run:

```bash
python scripts/generate_predictions.py \
  --provider openai-compatible \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name base-api-pagella-v0
```
