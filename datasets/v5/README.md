# Dataset v5

Dataset v5 is the P4 micro corrective dataset for FantaBrain LLM.

It is assembled as:

```text
datasets/v5/train.jsonl = datasets/v2/train.jsonl + 16 P4 examples
```

Dataset v3 and Dataset v4 are intentionally not used as a base. Qwen LoRA v2 remains the rollback baseline.

## P4 Blocks

- `mantra_no_modificatore`: 4 Mantra examples, no `modificatore` and no `reparto`.
- `classic_clean_vocab`: 4 Classic examples, no Mantra role-code language.
- `no_invented_modules`: 4 Mantra examples, use only modules named in the prompt.
- `italian_decision_clean`: 4 Classic examples, short decision-first answers with clean Italian.

All P4 rows use `source: v5_manual` and `quality_score: 5`.

Pagella v0 remains blind.
