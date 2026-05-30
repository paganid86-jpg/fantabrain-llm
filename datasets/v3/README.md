# Dataset v3

Dataset v3 is the P3 cleanup set for FantaBrain.

It is assembled as:

```text
datasets/v2/train.jsonl + datasets/v3/drafts/*.jsonl -> datasets/v3/train.jsonl
```

P3 adds 40 examples:

- 20 Mantra
- 20 Classic
- all `source: v3_manual`
- all `quality_score: 5`

## Blocks

1. `p3_block_001_mantra_no_module_invention.jsonl` - choose only among modules named by the user.
2. `p3_block_002_classic_modificatore_clean.jsonl` - clean Classic modifier explanations.
3. `p3_block_003_refusal_stop_clean.jsonl` - refusal that stops after grounded criteria and minimal data request.
4. `p3_block_004_mantra_roles_no_cross_mode.jsonl` - Mantra role-code reasoning without Classic leakage.
5. `p3_block_005_italiano_cleanup_decision_first.jsonl` - short, decision-first Italian.

## Authoring Rules

- Do not copy prompts or expected answers from `benchmarks/pagella_v0.jsonl`.
- Do not use real player names.
- Do not invent percentages, votes, scores, prices, modules, or rules.
- Mantra answers must use only modules named by the user, unless asking for missing module data.
- Classic answers must not use Mantra role-code logic.
- Refusal answers must not continue into invented specifics.
- First sentence must contain the decision or refusal.
- Keep assistant answers under 95 words.

## Validation

Run:

```bash
python scripts/assemble_dataset_v3.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v3/manifest.yaml \
  --output datasets/v3/train.jsonl
```

The command fails if a block is missing, counts are wrong, prompts are duplicated, quality is below 5, source is not `v3_manual`, or a P3 prompt exactly matches a pagella prompt.
