# OpenAI Fallback Eval v0

Questo runbook valuta offline il flusso app-style:

```text
Qwen/LoRA prediction -> output filter -> optional gpt-5.4-mini fallback -> output filter -> final answer
```

## Inputs

- Un run esistente con `predictions.jsonl`.
- `OPENAI_API_KEY` impostata come variabile ambiente o Colab secret.

Non committare API key, `.env`, token o credenziali.

## Command

```bash
python scripts/run_fallback_eval.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0-fallback-eval-v0 \
  --fallback-model gpt-5.4-mini \
  --max-output-tokens 350 \
  --temperature 0.2
```

## Outputs

- `fallback_eval.json`
- `fallback_eval.md`
- `fallback_predictions.jsonl`

## How To Read It

The most important fields are:

- `fallback_used_count`: how often Qwen/LoRA failed hard enough to need fallback.
- `fallback_success_count`: how often fallback cleared the second filter pass.
- `unresolved_safe_count`: how often both primary and fallback failed.
- `fallback_model`: actual model id returned by OpenAI for each fallback case.
- `estimated_total_cost_usd`: rough fallback cost from API usage tokens.

Cost estimation supports `gpt-5.4-mini` and versioned response ids such as
`gpt-5.4-mini-...`. If usage tokens are missing or the model is unknown, cost is
reported as `null` instead of guessed.

## Product Gate

This is promising only if fallback is used on a minority of cases, most fallback answers pass the second filter, unresolved safe cases are rare, and manual review confirms the fallback answers are not generic or misleading.
