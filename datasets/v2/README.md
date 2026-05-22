# Dataset v2

Dataset v2 is the P2 targeted repair set for FantaBrain.

It is assembled as:

```text
datasets/v1/train.jsonl + datasets/v2/drafts/*.jsonl -> datasets/v2/train.jsonl
```

P2 adds 80 examples:

- 40 Mantra
- 40 Classic
- all `source: v2_manual`
- all `quality_score: 5`

## Blocks

1. `p2_block_001_classic_modificatore.jsonl` - Classic modificatore, portiere, voto medio, floor.
2. `p2_block_002_mantra_role_codes_guardrail.jsonl` - literal Mantra role codes and module legality.
3. `p2_block_003_risk_varianza_decisioni.jsonl` - favorite/underdog, floor/upside, doubtful starters.
4. `p2_block_004_refusal_grounded_clean.jsonl` - refusal without invented certainty.
5. `p2_block_005_italiano_asciutto_decision_first.jsonl` - concise decision-first style.

## Authoring Rules

- Do not copy prompts or expected answers from `benchmarks/pagella_v0.jsonl`.
- Do not use real player names.
- Do not invent percentages, votes, scores, prices, or rules.
- Mantra answers must use role codes literally: `E`, `M`, `T`, `W`, `Pc`, `Dc`, `A`.
- Classic answers must use Classic vocabulary: porta, difesa, centrocampo, attacco, voto, bonus, malus, modificatore.
- First sentence must contain the decision or refusal.
- Keep assistant answers under 110 words.

## Validation

After `scripts/assemble_dataset_v2.py` is added, run:

```bash
python scripts/assemble_dataset_v2.py \
  --base datasets/v1/train.jsonl \
  --manifest datasets/v2/manifest.yaml \
  --output datasets/v2/train.jsonl
```

The command fails if a block is missing, counts are wrong, prompts are duplicated, quality is below 5, source is not `v2_manual`, or a P2 prompt exactly matches a pagella prompt.
