# Pagella v0 And Dataset v0 Design

## Goal

Build the first balanced evaluation and training set for the FantaBrain `coachino`, covering Fantacalcio Mantra and Fantacalcio Classic at the same level.

## Product Positioning

Pagella v0 and Dataset v0 are a proof-of-learning milestone, not a launch-grade App Store model milestone.

The first fine-tune on 120 examples should prove that the pipeline works, that the model can absorb FantaBrain tone, and that it can distinguish Mantra from Classic. A launch candidate should require a larger Dataset v1, a larger blind Pagella v1, provider fallback, backend-only integration, and privacy-safe handling of user data.

## Balance Decision

The v0 split is exactly 50/50:

- 20 Mantra pagella cases and 60 Mantra training examples.
- 20 Classic pagella cases and 60 Classic training examples.

Mantra and Classic must travel at the same quality level. The dataset must not make Mantra feel like the serious mode and Classic like a generic fallback.

## Pagella v0 Scope

Pagella v0 has 40 blind cases. These cases are not included in training.

| Task area | Total cases | Mantra | Classic |
| --- | ---: | ---: | ---: |
| Lineup advice | 8 | 4 | 4 |
| Auction advice | 8 | 4 | 4 |
| Trade / market advice | 8 | 4 | 4 |
| Rules / tactical explanation | 6 | 3 | 3 |
| Risk management | 6 | 3 | 3 |
| Refusal / grounding | 4 | 2 | 2 |
| **Total** | **40** | **20** | **20** |

Each case contains one user prompt, one ideal assistant answer, and tags that describe the intended skill. The eval report scores outputs from 1 to 5 on:

- mode correctness;
- tactical usefulness;
- groundedness;
- clarity;
- FantaBrain tone.

## Dataset v0 Scope

Dataset v0 has 120 supervised chat examples.

| Task area | Total examples | Mantra | Classic |
| --- | ---: | ---: | ---: |
| Lineup advice | 24 | 12 | 12 |
| Auction advice | 24 | 12 | 12 |
| Trade / market advice | 24 | 12 | 12 |
| Rules / tactical explanation | 18 | 9 | 9 |
| Risk management | 18 | 9 | 9 |
| Refusal / grounding | 12 | 6 | 6 |
| **Total** | **120** | **60** | **60** |

The source-of-truth v0 files should be versioned because they contain safe, manually written examples:

- `datasets/v0/train.jsonl`
- `benchmarks/pagella_v0.jsonl`

Generated splits and run artifacts remain ignored under `data/processed/`, `data/eval/`, and `reports/runs/`.

## Data Quality Rules

Every row must:

- use `mode: "mantra"` or `mode: "classic"`;
- use Italian user and assistant content;
- start messages with the shared FantaBrain system prompt;
- end messages with the assistant answer;
- contain one clear coaching task;
- include tags for mode, task, and skill;
- avoid private data, credentials, personal user data, or paid provider data;
- avoid live facts that can change, such as current injuries, real-time votes, updated lineups, or fresh calendar data;
- state when a decisive datum is missing instead of inventing it.

## Style Target

The ideal assistant is a private FantaBrain coach:

- practical and concise;
- tactical without sounding academic;
- confident when the prompt provides enough information;
- cautious when data is missing;
- explicit about Mantra role/slot/modulo logic;
- explicit about Classic reparto/bonus/modificatore logic.

## Error Handling

The preparation script should fail loudly when:

- a row is not valid JSONL;
- `mode` is not `mantra` or `classic`;
- messages do not include user and assistant turns;
- the first message is not `system`;
- the final message is not `assistant`;
- `quality_score`, when present, is outside 1 to 5.

## Verification

Local verification for v0 should include:

- Python compile check for `src`, `scripts`, and `tests`;
- dataset validation for `datasets/v0/train.jsonl`;
- pagella validation for `benchmarks/pagella_v0.jsonl`;
- deterministic preparation into ignored `data/processed/` and `data/eval/`;
- manual eval report generation from `benchmarks/pagella_v0.jsonl`.

## Next Milestone After v0

If the first forgia shows useful learning, Dataset v1 should target 500 to 800 high-quality examples and Pagella v1 should target 100 to 150 blind cases. That v1 milestone is the first credible path toward production discussion, not v0.
