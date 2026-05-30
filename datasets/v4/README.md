# Dataset v4

Dataset v4 is the P3bis corrective dataset for FantaBrain LLM.

It is assembled as:

```text
datasets/v4/train.jsonl = datasets/v2/train.jsonl + 20 P3bis examples
```

Dataset v3 is intentionally not used as a base because the v3 Pagella regressed versus v2.

## P3bis Blocks

- `mantra_anti_leak`: 5 Mantra examples, no `modificatore`.
- `classic_anti_leak`: 5 Classic examples, no Mantra vocabulary.
- `decision_inversion`: 6 examples, 3 Mantra and 3 Classic.
- `refusal_stop`: 4 examples, 2 Mantra and 2 Classic.

All P3bis rows use `source: v4_manual` and `quality_score: 5`.

Pagella v0 remains blind.
