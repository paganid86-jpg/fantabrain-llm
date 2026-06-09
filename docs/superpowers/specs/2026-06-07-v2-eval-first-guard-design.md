# V2 Eval-First Guard Design

## Context

`qwen25-3b-fantabrain-sft-v2` is the current champion adapter. Later adapters improved or regressed isolated metrics, but none beat v2 on the promotion gates. V5 improved hallucination-free count, yet still failed the most important lexical and grounding checks:

- case 2 invented Mantra modules;
- Mantra still leaked `modificatore` / `modificatori`;
- Classic still leaked Mantra-style role and module language;
- malformed Italian remained recurring.

The next experiment should therefore be eval-first, not another training pass. The goal is to learn whether prompt fencing and automatic output audits can improve v2 behavior before creating more data.

## Goal

Add a repeatable evaluation path that runs the v2 adapter with a stricter prompt guard and then audits the generated predictions for mode leakage, invented modules, malformed Italian, and known P4 failure patterns.

The experiment should answer one question: can v2 become safer with inference-time controls alone?

## Non-Goals

- Do not create Dataset v6.
- Do not train or modify a LoRA adapter.
- Do not change Pagella v0 cases.
- Do not post-process model answers into safer final text yet. This experiment measures model behavior under a guard prompt; app-level filtering can be designed later.

## Prompt Guard Design

The guarded run should keep the original eval case messages but strengthen the system instruction at inference time. The safest shape is to merge the guard into the existing system message before generation, producing one system message rather than multiple system messages.

The first guard preset is `mode_fence_v1`.

Shared rules:

- Answer in clean Italian.
- Start with the decision or refusal.
- Do not invent player names, live facts, exact future votes, or unavailable probabilities.
- If key context is missing, say what is missing and give only a general criterion.
- Avoid malformed invented words.

Mantra rules:

- Reason with role codes, slot coverage, module constraints, and panchina compatibility.
- Do not use `modificatore`, `modificatori`, or `reparto`.
- If the user mentions specific modules, do not introduce extra modules.
- If the user does not mention modules, do not invent a module number.

Classic rules:

- Reason with reparti, titolarita, bonus, malus, modificatore difesa when relevant, and panchina by reparto.
- Do not use Mantra-only role codes such as `Pc`, `T`, `W`, `A`, `M`, `E`, `Dc`, `Dd`, or `Ds` unless the user explicitly asks about Mantra.
- Do not talk about module incastri as if Classic were Mantra.

## Prediction Audit Design

Add an audit that reads `predictions.jsonl` and reports violations per case. The audit should be deterministic and text-based so it can run locally without a model.

Checks:

- `mantra_forbidden_terms`: Mantra predictions must not contain `modificatore`, `modificatori`, or `reparto`.
- `classic_role_code_leakage`: Classic predictions must not contain standalone Mantra role codes unless present in the prompt.
- `invented_modules`: if a Mantra prompt mentions module numbers, the prediction must not mention module numbers outside that prompt set.
- `classic_module_language`: Classic predictions should flag terms like `slot`, `incastri`, `moduli`, and `modulo principale`.
- `malformed_terms`: flag known recurring malformed terms from v4/v5, including `offENSIVO`, `multiruomo`, `multiruoco`, `sicurata`, `attaccantini`, `attorcicati`, `inattaccante`, `punteggia`, `punteggiere`, `migliro`, and `voti esattissimi`.

The audit should produce:

- a JSON report for automation;
- a Markdown report for manual review;
- a summary count by check name;
- a non-zero exit option for hard gates.

## CLI Shape

Extend prediction generation with an optional prompt guard:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4 \
  --load-in-4bit \
  --torch-dtype float16 \
  --prompt-guard mode_fence_v1
```

Then run:

```bash
python scripts/audit_predictions.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0 \
  --fail-on-hard-gates
```

## Colab Notebook Direction

The notebook should stay minimal:

1. GPU and token check.
2. Clone or update repo from `master`.
3. Install dependencies.
4. Restore `qwen25-3b-fantabrain-sft-v2-adapter.zip`.
5. Run vanilla v2 only if the user wants a fresh comparison.
6. Run guarded v2 with `--prompt-guard mode_fence_v1`.
7. Run prediction audit.
8. Zip and download/copy the guarded report.

## Success Criteria

The guarded v2 run is promising only if it beats or preserves the v2 baseline while reducing hard violations:

- effective average should be greater than or equal to 2.69 after manual scoring;
- raw average should stay at least 3.10;
- hallucination-free should stay at least 26/40;
- case 2 should not invent modules;
- Mantra predictions should have zero `modificatore` / `modificatori`;
- Classic predictions should reduce role-code and module-language leakage;
- malformed Italian should be lower than v5.

If the guarded run fails these gates, v2 vanilla remains the champion and the next step should be app-level output filtering or a different base/model strategy, not another blind dataset expansion.

## Testing

The implementation should add focused unit tests for:

- prompt guard injection keeps one system message and preserves user/assistant order;
- `mode_fence_v1` adds Mantra rules for Mantra examples and Classic rules for Classic examples;
- module extraction catches invented module numbers without false positives like `3-4-2` inside `3-4-2-1`;
- role-code leakage checks are token-aware;
- audit JSON and Markdown outputs include case ids and summary counts;
- `generate_predictions.py --prompt-guard mode_fence_v1` still works with the echo provider for a cheap local smoke test.

## Rollback

This is eval-only. If it fails, remove or ignore the guarded run artifacts and keep using `qwen25-3b-fantabrain-sft-v2` as the current baseline.
